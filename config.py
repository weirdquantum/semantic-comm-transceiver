import os
import torch


class Config:
    # ----- paths -----
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    result_dir = os.path.join(base_dir, "results")

    # ----- vocabulary & tokens -----
    PAD_TOKEN = "<PAD>"
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    START_IDX = 1
    END_IDX = 2
    UNK_IDX = 3
    vocab_size = 10000
    max_seq_len = 32
    min_seq_len = 4

    # ----- model architecture -----
    d_model = 128
    n_heads = 8
    d_ff = 512
    n_encoder_layers = 4
    n_decoder_layers = 4
    channel_dim = 16
    dropout = 0.1

    # ----- training -----
    batch_size = 256
    epochs = 80
    lr = 1e-3
    lr_min = 1e-5
    weight_decay = 1e-5
    grad_clip = 1.0
    snr_train_low = -10.0   # dB
    snr_train_high = 20.0   # dB
    warmup_epochs = 5

    # ----- composite loss weights -----
    alpha_ce = 1.0
    beta_semantic = 0.2

    # ----- evaluation SNR points -----
    eval_snr_list = [-10, -5, 0, 5, 10, 15, 20]

    # ----- channel types -----
    AWGN = "awgn"
    RAYLEIGH = "rayleigh"

    # ----- device -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.data_dir, cls.checkpoint_dir, cls.result_dir]:
            os.makedirs(d, exist_ok=True)
