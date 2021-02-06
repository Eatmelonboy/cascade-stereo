#!/usr/bin/env bash
export NGPUS=1
export CUDA_VISIBLE_DEVICES=5
python  train_consistency_maskloss.py  --batch_size=1 --logdir='./checkpoints/test'  ${@:3}