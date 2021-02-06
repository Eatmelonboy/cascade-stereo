import argparse, os, sys, time, gc, datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import torch.distributed as dist

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='A PyTorch Implementation of Cascade Cost Volume MVSNet')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--device', default='cuda', help='select model')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', default='/media/public/disk0/zzy/mvsnet/training_data/dtu_training/',help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', default='./lists/dtu/train.txt',help='train list')
parser.add_argument('--testlist', default='./lists/dtu/test.txt',help='test list')

parser.add_argument('--epochs', type=int, default=13, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/usp', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=50, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--pin_m', action='store_true', help='data loader pin memory')
parser.add_argument("--local_rank", type=int, default=0)

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')
parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')#每个像素深度采样个数的变化。。mvsnet192个，本网络第一层采样48个，所以下面的采样间隔是mvsnet的4倍。
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')#深度采样值间隔是mvsnet的4，2，1倍
parser.add_argument('--dlossw', type=str, default="0.5,1.0,2.0", help='depth loss weight for different stage')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')

parser.add_argument('--using_apex', action='store_true', help='using apex, need to install apex')
parser.add_argument('--sync_bn', action='store_true',help='enabling apex sync BN.')
parser.add_argument('--opt-level', type=str, default="O0")
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)


num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1

# main function
def train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args):
    milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0/3, warmup_iters=500,
                                                        last_epoch=len(TrainImgLoader) * start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        global_step = len(TrainImgLoader) * epoch_idx

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(model, model_loss, optimizer, sample, args)
            lr_scheduler.step()
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    save_images(logger, 'train', image_outputs, global_step)
                    print(
                       "Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss = {:.3f}, depth loss = {:.3f}, time = {:.3f}".format(
                           epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                           optimizer.param_groups[0]["lr"], loss,
                           scalar_outputs['depth_loss'],
                           time.time() - start_time))
                del scalar_outputs, image_outputs

        # checkpoint
        if (not is_distributed) or (dist.get_rank() == 0):
            if (epoch_idx + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        if (epoch_idx % args.eval_freq == 0) or (epoch_idx == args.epochs - 1):
            avg_test_scalars = DictAverageMeter()
            for batch_idx, sample in enumerate(TestImgLoader):
                start_time = time.time()
                global_step = len(TrainImgLoader) * epoch_idx + batch_idx
                do_summary = global_step % args.summary_freq == 0
                loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args)
                if (not is_distributed) or (dist.get_rank() == 0):
                    if do_summary:
                        save_scalars(logger, 'test', scalar_outputs, global_step)
                        save_images(logger, 'test', image_outputs, global_step)
                        print("Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, depth loss = {:.3f}, time = {:3f}".format(
                                                                            epoch_idx, args.epochs,
                                                                            batch_idx,
                                                                            len(TestImgLoader), loss,
                                                                            scalar_outputs["depth_loss"],
                                                                            time.time() - start_time))
                    avg_test_scalars.update(scalar_outputs)
                    del scalar_outputs, image_outputs

            if (not is_distributed) or (dist.get_rank() == 0):
                save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
                print("avg_test_scalars:", avg_test_scalars.mean())
            gc.collect()


def test(model, model_loss, TestImgLoader, args):
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        if (not is_distributed) or (dist.get_rank() == 0):
            print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                        time.time() - start_time))
            if batch_idx % 100 == 0:
                print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    if (not is_distributed) or (dist.get_rank() == 0):
        print("final", avg_test_scalars.mean())



def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    
    proj = torch.matmul(src_proj, torch.inverse(ref_proj))
    rot = proj[:, :3, :3]  # [B,3,3]
    trans = proj[:, :3, 3:4]  # [B,3,1]

    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                            torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
    rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
    rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                        -1)  # [B, 3, Ndepth, H*W]
    proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
    proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
    proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
    proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
    proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
    grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

def compute_reprojection_loss( pred, target):
    """Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean()


    return l1_loss

def train_sample(model, model_loss, optimizer, sample, args):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]#(1,512,640)

    ###

    proj_matrices = sample_cuda["proj_matrices"]["stage3"]
    proj_matrices = torch.unbind(proj_matrices, 1)

    ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

    imgs=[]
    for nview_idx in range(sample_cuda["imgs"].size(1)):  # imgs shape (B, N, C, H, W)
        img = sample_cuda["imgs"][:, nview_idx]
        imgs.append(img)
    ref_img=imgs[0]
    source_img=imgs[1:]

    warpimgs=[]

    for src_img, src_proj in zip(source_img, src_projs):#将不同src的feature和投影矩阵配对
        #warpped features
        src_proj_new = src_proj[:, 0].clone()
        src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
        ref_proj_new = ref_proj[:, 0].clone()
        ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
        for i in range(1,4):
            depth_est = outputs['stage{}'.format(i)]['depth']
            depth = depth_est.unsqueeze(1)
            depth = F.interpolate(depth,
                                      [src_img.shape[2], src_img.shape[3]], mode='bilinear')
            warped_image = homo_warping(src_img, src_proj_new, ref_proj_new, depth)
            warped_image=warped_image.squeeze(2)
            warpimgs.append(warped_image)
    loss=0
    mask=mask.unsqueeze(1).repeat(1,3,1,1)
    mask = mask > 0.5

    for i in warpimgs:
        #print((i - ref_img).abs().shape)
        loss+=compute_reprojection_loss(i[mask] , ref_img[mask])

        # loss += F.smooth_l1_loss(i[mask], ref_img[mask], reduction='mean')

    #loss=loss/2

    depth_loss=loss
    if is_distributed and args.using_apex:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()

    scalar_outputs = {"loss": loss,
                      'depth_loss':depth_loss,
                      }
    mask = mask_ms["stage{}".format(num_stage)]
    image_outputs = {"depth_est": depth_est * mask,
                     'warp image1':warpimgs[0],
                     'warp image2': warpimgs[1],
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"]["stage1"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage1"],
                     "errormap": (depth_est - depth_gt).abs() * mask,
                     }

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


@make_nograd_func
def test_sample_depth(model, model_loss, sample, args):
    if is_distributed:
        model_eval = model.module
    else:
        model_eval = model
    model_eval.eval()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model_eval(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    ###

    proj_matrices = sample_cuda["proj_matrices"]["stage3"]
    proj_matrices = torch.unbind(proj_matrices, 1)

    ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

    imgs = []
    for nview_idx in range(sample_cuda["imgs"].size(1)):  # imgs shape (B, N, C, H, W)
        img = sample_cuda["imgs"][:, nview_idx]
        imgs.append(img)
    ref_img = imgs[0]
    source_img = imgs[1:]

    warpimgs = []

    for src_img, src_proj in zip(source_img, src_projs):  # 将不同src的feature和投影矩阵配对
        # warpped features
        src_proj_new = src_proj[:, 0].clone()
        src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
        ref_proj_new = ref_proj[:, 0].clone()
        ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
        for i in range(1, 4):
            depth_est = outputs['stage{}'.format(i)]['depth']
            depth = depth_est.unsqueeze(1)
            depth = F.interpolate(depth,
                                      [src_img.shape[2], src_img.shape[3]], mode='bilinear')
            warped_image = homo_warping(src_img, src_proj_new, ref_proj_new, depth)
            warped_image = warped_image.squeeze(2)
            warpimgs.append(warped_image)
    loss = 0
    mask = mask.unsqueeze(1).repeat(1, 3, 1, 1)
    mask = mask > 0.5

    for i in warpimgs:
        # loss += F.smooth_l1_loss(i[mask], ref_img[mask], reduction='mean')
        loss+=compute_reprojection_loss(i[mask] , ref_img[mask])


    depth_loss = loss


    scalar_outputs = {"loss": loss,
                      'depth_loss': depth_loss,
                    }
    mask = mask_ms["stage{}".format(num_stage)]
    image_outputs = {"depth_est": depth_est * mask,
                     "depth_est_nomask": depth_est,
                     'warp image1': warpimgs[0],
                     'warp image2': warpimgs[1],
                     "depth_gt": sample["depth"]["stage1"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage1"],
                     "errormap": (depth_est - depth_gt).abs() * mask}

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)

def profile():
    warmup_iter = 5
    iter_dataloader = iter(TestImgLoader)

    @make_nograd_func
    def do_iteration():
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        test_sample(next(iter_dataloader), detailed_summary=True)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return end_time - start_time

    for i in range(warmup_iter):
        t = do_iteration()
        print('WarpUp Iter {}, time = {:.4f}'.format(i, t))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for i in range(5):
            t = do_iteration()
            print('Profile Iter {}, time = {:.4f}'.format(i, t))
            time.sleep(0.02)

    if prof is not None:
        # print(prof)
        trace_fn = 'chrome-trace.bin'
        prof.export_chrome_trace(trace_fn)
        print("chrome trace file is written to: ", trace_fn)


if __name__ == '__main__':
    # parse arguments and check
    args = parser.parse_args()

    # using sync_bn by using nvidia-apex, need to install apex.
    if args.sync_bn:
        assert args.using_apex, "must set using apex and install nvidia-apex"
    if args.using_apex:
        try:
            from apex.parallel import DistributedDataParallel as DDP
            from apex.fp16_utils import *
            from apex import amp, optimizers
            from apex.multi_tensor_apply import multi_tensor_applier
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

    if args.resume:
        assert args.mode == "train"
        assert args.loadckpt is None
    if args.testpath is None:
        args.testpath = args.trainpath

    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    set_random_seed(args.seed)
    device = torch.device(args.device)

    if (not is_distributed) or (dist.get_rank() == 0):
        # create logger for mode "train" and "testall"
        if args.mode == "train":
            if not os.path.isdir(args.logdir):
                os.makedirs(args.logdir)
            current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            print("current time", current_time_str)
            print("creating new summary file")
            logger = SummaryWriter(args.logdir)
        print("argv:", sys.argv[1:])
        print_args(args)

    # model, optimizer  ##ndepths：48，32，8 interal_ratio:4,2,1
    model = CascadeMVSNet(refine=False, ndepths=[48,32,8],
                          depth_interals_ratio=[2.0,2.0,1.0],
                          share_cr=False,
                          cr_base_chs=[8,8,8],
                          grad_method="detach")
    model.to(device)
    model_loss = cas_mvsnet_loss

    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

    # load parameters
    start_epoch = 0
    if args.resume:
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    elif args.loadckpt:
        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])

    if (not is_distributed) or (dist.get_rank() == 0):
        print("start at epoch {}".format(start_epoch))
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.using_apex:
        # Initialize Amp
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                          loss_scale=args.loss_scale
                                          )

    if is_distributed:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # find_unused_parameters=False,
            # this should be removed if we update BatchNorm stats
            # broadcast_buffers=False,
        )
    else:
        if torch.cuda.is_available():
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 3, args.numdepth, args.interval_scale)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", 5, args.numdepth, args.interval_scale)

    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                            rank=dist.get_rank())
        test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=dist.get_world_size(),
                                                           rank=dist.get_rank())

        TrainImgLoader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, num_workers=1,
                                    drop_last=True,
                                    pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler, num_workers=1, drop_last=False,
                                   pin_memory=args.pin_m)
    else:
        TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=1, drop_last=True,
                                    pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=1, drop_last=False,
                                   pin_memory=args.pin_m)


    if args.mode == "train":
        train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args)
    elif args.mode == "test":
        test(model, model_loss, TestImgLoader, args)
    elif args.mode == "profile":
        profile()
    else:
        raise NotImplementedError