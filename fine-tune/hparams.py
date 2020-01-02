import tensorflow as tf
#from text import symbols

def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=60,
        iters_per_checkpoint=100,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,

        ################################
        # Data Parameters              #
        ################################
        training_list='/home/jxzhang/Documents/DataSets/cmu_us_slt_arctic-0.95-release/list/train_non-parallel_slt_rms.list',
        validation_list='/home/jxzhang/Documents/DataSets/cmu_us_slt_arctic-0.95-release/list/eval_slt_rms.list',
        mel_mean_std='/home/jxzhang/Documents/DataSets/VCTK/mel_mean_std.npy',

        speaker_A='slt',
        speaker_B='rms',
        a_embedding_path='zero_embeddings.npy',
        b_embedding_path='zero_embeddings.npy',
        ################################
        # Data Parameters              #
        ################################
        n_mel_channels=80,
        n_spc_channels=1025,
        n_symbols=41, #
        pretrain_n_speakers=99, #
        n_speakers=2,
        predict_spectrogram=False,

        ################################
        # Model Parameters             #
        ################################

        symbols_embedding_dim=512,

        # Text Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,
        text_encoder_dropout=0.5,

        # Audio Encoder parameters
        spemb_input=False,
        n_frames_per_step_encoder=2,  
        audio_encoder_hidden_dim=512,
        AE_attention_dim=128,
        AE_attention_location_n_filters=32,
        AE_attention_location_kernel_size=51,
        beam_width=10,

        # hidden activation 
        # relu linear tanh
        hidden_activation='tanh',

        #Speaker Encoder parameters
        speaker_encoder_hidden_dim=256,
        speaker_encoder_dropout=0.2,
        speaker_embedding_dim=128,

        #Text Classifier parameters
        #text_classifier_hidden_dim=256,

        #Speaker Classifier parameters
        SC_hidden_dim=512,
        SC_n_convolutions=3,
        SC_kernel_size=5,

        # Decoder parameters
        feed_back_last=True,
        n_frames_per_step_decoder=2,
        decoder_rnn_dim=512,
        prenet_dim=[256,256],
        max_decoder_steps=1000,
        gate_threshold=0.5,
    
        # Attention parameters
        attention_rnn_dim=512,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=17,

        # PostNet parameters
        postnet_n_convolutions=5,
        postnet_dim=512,
        postnet_kernel_size=5,
        postnet_dropout=0.5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=5.0,
        batch_size=8,
        warmup=7,
        decay_rate=0.5,
        decay_every=7,
        
        contrastive_loss_w=30.0,
        consistent_loss_w=0.0,
        speaker_encoder_loss_w=0.,
        text_classifier_loss_w=1.0,
        speaker_adversial_loss_w=0.2,
        speaker_classifier_loss_w=1.0,

    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', list(hparams.values()))

    return hparams
