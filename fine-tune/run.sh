# slt to rms, rms to slt #
RUN_EMB=true
RUN_TRAIN=true
RUN_GEN=true

export CUDA_VISIBLE_DEVICES=2
speaker_A='slt'
speaker_B='rms'
training_list="/home/jxzhang/Documents/DataSets/cmu_us_slt_arctic-0.95-release/list/train_non-parallel_${speaker_A}_${speaker_B}.list"
validation_list="/home/jxzhang/Documents/DataSets/cmu_us_slt_arctic-0.95-release/list/eval_${speaker_A}_${speaker_B}.list"

logdir="logdir_${speaker_A}_${speaker_B}"
pretrain_checkpoint_path='../pre-train/outdir/checkpoint_0'
finetune_ckpt="checkpoint_100"

contrastive_loss_w=30.0
speaker_adversial_loss_w=0.2
speaker_classifier_loss_w=1.0
decay_every=7
warmup=7
epochs=70
batch_size=8
SC_kernel_size=1
learning_rate=1e-3
gen_num=66

if $RUN_EMB
then
    echo 'running embeddings...'
    python inference_embedding.py \
    -c $pretrain_checkpoint_path \
    --hparams=speaker_A=$speaker_A,\
speaker_B=$speaker_B,\
training_list=${training_list},SC_kernel_size=$SC_kernel_size
fi

if $RUN_TRAIN
then
    echo 'running trainings...'
    python train.py  \
        -l $logdir -o outdir --n_gpus=1 \
        -c $pretrain_checkpoint_path \
        --warm_start \
        --hparams=speaker_A=$speaker_A,\
speaker_B=$speaker_B,a_embedding_path="outdir/embeddings/${speaker_A}.npy",\
b_embedding_path="outdir/embeddings/${speaker_B}.npy",\
training_list=$training_list,\
validation_list=$validation_list,\
contrastive_loss_w=$contrastive_loss_w,\
speaker_adversial_loss_w=$speaker_adversial_loss_w,\
speaker_classifier_loss_w=$speaker_classifier_loss_w,\
decay_every=$decay_every,\
epochs=$epochs,\
warmup=$warmup,batch_size=$batch_size,\
SC_kernel_size=$SC_kernel_size,learning_rate=$learning_rate
fi


if $RUN_GEN
then 
    echo 'running generations...'
    python inference.py \
        -c outdir/$logdir/$finetune_ckpt \
        --num $gen_num \
        --hparams=validation_list=$validation_list,SC_kernel_size=$SC_kernel_size
fi
