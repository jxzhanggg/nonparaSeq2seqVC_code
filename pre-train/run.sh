#!/bin/bash

# you can set the hparams by using --hparams=xxx
CUDA_VISIBLE_DEVICES=3 python train.py -l logdir \
-o outdir --n_gpus=1 --hparams=speaker_adversial_loss_w=20.,ce_loss=False,speaker_classifier_loss_w=0.1,contrastive_loss_w=30.