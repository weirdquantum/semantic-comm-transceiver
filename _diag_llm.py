import torch
from config import Config
from dataset import get_dataloader, load_vocab
from train import build_model
from utils import batch_ids_to_texts
from models.llm_refiner import LLMRefiner

device = Config.device
w2i, i2w = load_vocab()
vocab_size = len(w2i)

model = build_model("improved", vocab_size, device)
ckpt = torch.load("checkpoints/improved_awgn/best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

refiner = LLMRefiner(device=device)

loader = get_dataloader("test", 64, shuffle=False)
batch = next(iter(loader)).to(device)
ref_texts = batch_ids_to_texts(batch, i2w, 2)

for snr_val in [-10.0, 0.0, 10.0]:
    snr_db = torch.full((batch.size(0),), snr_val, device=device, dtype=torch.float)
    pred_ids = model.greedy_decode(batch, snr_db, "awgn", max_len=32, start_idx=1, end_idx=2)
    pred_texts = batch_ids_to_texts(pred_ids, i2w, 2)

    sample_pred = pred_texts[:10]
    sample_ref = ref_texts[:10]
    refined = refiner.refine(sample_pred)

    n_changed = sum(1 for a, b in zip(sample_pred, refined) if a != b)
    print(f"\n=== SNR={snr_val:.0f} dB | changed {n_changed}/{len(sample_pred)} ===")
    for j in range(min(5, len(sample_pred))):
        changed = sample_pred[j] != refined[j]
        ppl = refiner._sentence_ppl(sample_pred[j])
        tag = " <<FIXED>>" if changed else ""
        print(f"  REF : {sample_ref[j]}")
        print(f"  PRED: {sample_pred[j]}  (ppl={ppl:.0f}){tag}")
        if changed:
            print(f"  LLM : {refined[j]}")
        print()
