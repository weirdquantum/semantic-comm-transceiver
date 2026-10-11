"""Fast sanity checks of the model code (no data or GPU needed): `pytest`."""

import pytest
import torch

from models import BaselineDeepSC, ImprovedDeepSC
from models.channel import rayleigh
from models.modules import ChannelEncoder

V, L, B = 60, 12, 16          # vocab size, sequence length, batch size
PAD, START, END = 0, 1, 2


def _batch(seed=0):
    """Random padded sentences: START w1 ... wn END PAD ..."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros(B, L, dtype=torch.long)
    for i, n in enumerate(torch.randint(3, L - 2, (B,), generator=g).tolist()):
        ids[i, 0] = START
        ids[i, 1:n + 1] = torch.randint(4, V, (n,), generator=g)
        ids[i, n + 1] = END
    return ids


def _model(cls, **kwargs):
    torch.manual_seed(0)
    return cls(V, d_model=32, n_heads=4, d_ff=64, n_enc_layers=2,
               n_dec_layers=2, max_len=L, **kwargs).eval()


def _until_end(seq):
    seq = seq.tolist()
    return seq[:seq.index(END) + 1] if END in seq else seq


def test_rayleigh_has_unit_average_gain():
    torch.manual_seed(0)
    _, snr_eff = rayleigh(torch.zeros(100_000, 1, 1), torch.zeros(100_000))
    assert abs((10 ** (snr_eff / 10)).mean().item() - 1.0) < 0.02


def test_masked_power_spends_the_budget_on_valid_tokens():
    ids = _batch()
    out = ChannelEncoder(32, 8, masked_power=True)(torch.randn(B, L, 32), valid_mask=ids != PAD)
    energy = (out ** 2).sum(dim=(1, 2))
    assert torch.allclose(energy, torch.full_like(energy, L * 8), rtol=1e-4)
    assert out[ids == PAD].abs().max() == 0


def test_baseline_has_no_snr_parameters():
    names = [n for n, _ in _model(BaselineDeepSC).named_parameters()]
    assert not any(k in n for n in names for k in ("snr_emb", "caq", "film"))


def test_film_is_the_identity_at_initialisation():
    ids, snr = _batch(), torch.zeros(B)
    with_film = _model(ImprovedDeepSC)
    without = _model(ImprovedDeepSC, use_film=False)
    without.load_state_dict(with_film.state_dict(), strict=False)
    torch.manual_seed(1)
    a = with_film(ids, ids[:, :-1], snr)[0]
    torch.manual_seed(1)
    b = without(ids, ids[:, :-1], snr)[0]
    assert torch.equal(a, b)


@pytest.mark.parametrize("cls", [BaselineDeepSC, ImprovedDeepSC])
@pytest.mark.parametrize("channel", ["awgn", "rayleigh"])
def test_beam_width_one_equals_greedy(cls, channel):
    model, ids, snr = _model(cls), _batch(), torch.full((B,), 5.0)
    torch.manual_seed(2)
    greedy, _ = model.decode(ids, snr, channel, max_len=L)
    torch.manual_seed(2)
    beam = model._beam(*model._transmit(ids, snr, channel)[1:], L, START, END, 1)
    assert [_until_end(s) for s in greedy] == [_until_end(s) for s in beam]


def test_decode_returns_consistent_candidates():
    model, ids = _model(ImprovedDeepSC), _batch()
    ys, info = model.decode(ids, torch.zeros(B), "awgn", max_len=L, beam_width=3, topk=5)
    assert (ys[:, 0] == START).all()
    assert info["cand_ids"].shape == (B, ys.size(1) - 1, 5)
    # the chosen token can never be more likely than the best candidate
    assert (info["conf"] <= info["cand_logp"][..., 0].exp() + 1e-6).all()
