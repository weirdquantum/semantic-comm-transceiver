import os
import torch


class Config:
    # ----- paths -----
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    # SC_CKPT_DIR / SC_RESULT_DIR let cloud runs write to persistent storage
    checkpoint_dir = os.environ.get("SC_CKPT_DIR", os.path.join(base_dir, "checkpoints"))
    result_dir = os.environ.get("SC_RESULT_DIR", os.path.join(base_dir, "results"))
    bert_dir = os.path.join(checkpoint_dir, "bert_europarl")

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
    val_fraction = 0.05     # carved out of the training split
    val_snr_list = [-5.0, 0.0, 5.0]   # model selection SNRs (dB)

    # ----- loss weights -----
    alpha_ce = 1.0
    beta_feat = 0.1         # feature-denoising loss weight (improved model only)

    # ----- evaluation SNR points -----
    eval_snr_list = [-10, -5, 0, 5, 10, 15, 20]
    eval_max_samples = 10000

    # ----- LLM fusion refiner (tuned on the validation split) -----
    llm_topk = 5
    llm_lambda_grid = [0.0, 0.25, 0.5, 1.0, 2.0]
    llm_tau_grid = [0.5, 0.7, 0.9]
    llm_tune_snrs = [-5.0, 0.0, 5.0]
    llm_tune_samples = 2000

    # ----- device -----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
