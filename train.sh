#!/usr/bin/env bash
export NGPUS=1
export CUDA_VISIBLE_DEVICES=1
python  train.py  --batch_size=4 \
# --loadckpt='./checkpoints/usp/model_000000.ckpt' \
--logdir='./checkpoints/debugnew'