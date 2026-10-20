import torch

from lsa.models.decoder import Decoder, build_decoder
from lsa.models.latent import LatentTable


def test_decoder_shapes_and_skip_width():
    dec = Decoder(latent_dim=128, hidden_dim=512, n_layers=8, skip_layers=(4,))
    assert len(dec.hidden) == 8
    # layer 3 shrinks so that concat([h, z, x]) is exactly 512 wide at layer 4
    assert dec.hidden[3].weight.shape == (512 - 131, 512)
    assert dec.hidden[4].weight.shape == (512, 512)
    assert dec.hidden[0].weight.shape == (512, 131)
    assert dec.out.weight.shape == (1, 512)
    z, x = torch.randn(2, 128), torch.rand(2, 50, 3)
    assert dec(z, x).shape == (2, 50)
    # per-point latents give the same result as broadcast latents
    assert torch.allclose(dec(z[:, None].expand(-1, 50, -1), x), dec(z, x))


def test_decoder_weight_norm_parametrized():
    dec = Decoder(latent_dim=8, hidden_dim=32, n_layers=8)
    names = dict(dec.named_parameters())
    assert "hidden.0.parametrizations.weight.original0" in names  # g
    assert "hidden.0.parametrizations.weight.original1" in names  # v
    assert "out.weight" in names  # output layer is plain linear
    plain = Decoder(latent_dim=8, hidden_dim=32, weight_norm=False)
    assert "hidden.0.weight" in dict(plain.named_parameters())


def test_decoder_small_hidden_skip_fallback():
    dec = Decoder(latent_dim=64, hidden_dim=32, n_layers=6, skip_layers=(3,))
    assert dec(torch.randn(1, 64), torch.rand(1, 7, 3)).shape == (1, 7)


def test_decoder_gradients_reach_latents():
    dec = build_decoder({"latent_dim": 16, "hidden_dim": 64})
    z = torch.randn(3, 16, requires_grad=True)
    dec(z, torch.rand(3, 20, 3)).sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_sdf_fn_matches_forward():
    dec = Decoder(latent_dim=16, hidden_dim=64)
    z, x = torch.randn(16), torch.rand(30, 3)
    assert torch.allclose(dec.sdf_fn(z)(x), dec(z[None], x[None])[0])


def test_latent_table_init():
    lat = LatentTable(2000, 128, init_std=0.01)
    assert lat.weight.shape == (2000, 128)
    assert abs(lat.weight.std().item() - 0.01) < 5e-4
    assert abs(lat.weight.mean().item()) < 5e-4
    assert lat(torch.tensor([0, 5])).shape == (2, 128)
