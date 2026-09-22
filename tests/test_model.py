"""Regression tests for the core TarFlow model (tarflow/model.py)."""

from __future__ import annotations

import pytest
import torch

from tarflow import TarFlow
from tarflow.model import (
    Attention,
    MetaBlock,
    PermutationFlip,
    PermutationIdentity,
    get_tarflow_model,
)

B, T, D = 2, 6, 4  # batch, num_tokens, token_size
PROJ = 64  # must be divisible by the attention head_dim (64)
NUM_CLASSES, COND_DIM = 3, 5

# model kwargs for each conditioning/flow variant supported by the model
VARIANTS: dict[str, dict] = {
    "uncond": {},
    "vp": {"nvp": False},
    "class": {"num_classes": NUM_CLASSES},
    "continuous": {"cond_dim": COND_DIM},
}


def fill_params(model: torch.nn.Module, seed: int = 0) -> None:
    """Set parameters deterministically, independent of construction order."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(model.named_parameters()):
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)


def make_model(variant: str, num_blocks: int = 3, layers: int = 2) -> TarFlow:
    model = TarFlow(T, D, PROJ, num_blocks, layers, **VARIANTS[variant]).eval()
    fill_params(model)
    return model


def make_inputs(variant: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(B, T, D, generator=g)
    if variant == "class":
        return x, torch.tensor([0, 2])
    if variant == "continuous":
        return x, torch.randn(B, COND_DIM, generator=g)
    return x, None


@pytest.fixture(params=[True, False], ids=["spda", "base_attn"])
def use_spda(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> bool:
    monkeypatch.setattr(Attention, "USE_SPDA", request.param)
    return request.param


# ---------------------------------------------------------------- forward


@pytest.mark.parametrize("variant", VARIANTS)
def test_forward_shapes(variant: str) -> None:
    model = make_model(variant)
    x, y = make_inputs(variant)
    z, outputs, logdets = model(x, y)
    assert z.shape == x.shape
    assert logdets.shape == (B,)
    assert len(outputs) == len(model.blocks)
    assert all(o.shape == x.shape for o in outputs)
    assert outputs[-1] is z


# (z.sum(), z[0, -1, :2], logdets) for `make_model(variant)(*make_inputs(variant))`
GOLDEN = {
    "uncond": (-8.058834, [0.769042, 0.148099], [-0.013332, -0.047412]),
    "vp": (-3.719223, [0.848252, 0.092289], [0.0, 0.0]),
    "class": (-7.271933, [0.916729, 0.274892], [0.032200, 0.014596]),
    "continuous": (-11.225681, [0.655062, 0.060481], [0.010428, -0.053693]),
}


@pytest.mark.parametrize("variant", VARIANTS)
def test_forward_golden_values(variant: str, use_spda: bool) -> None:
    """Numerical output of the forward pass must not drift."""
    model = make_model(variant)
    with torch.no_grad():
        z, _, logdets = model(*make_inputs(variant))
    z_sum, z_last, expected_logdets = GOLDEN[variant]
    kwargs = {"atol": 1e-4, "rtol": 1e-4}
    torch.testing.assert_close(z.sum(), torch.tensor(z_sum), **kwargs)
    torch.testing.assert_close(z[0, -1, :2], torch.tensor(z_last), **kwargs)
    torch.testing.assert_close(logdets, torch.tensor(expected_logdets), **kwargs)


@pytest.mark.parametrize("variant", ["uncond", "vp"])
def test_logdet_matches_jacobian(variant: str) -> None:
    """logdets is log|det J| per dimension (i.e. divided by T * D)."""
    model = make_model(variant)
    x, _ = make_inputs(variant)
    x = x[:1]
    jac = torch.autograd.functional.jacobian(lambda x_: model(x_)[0], x)
    _, expected = torch.linalg.slogdet(jac.reshape(T * D, T * D))
    _, _, logdets = model(x)
    torch.testing.assert_close(logdets[0], expected / (T * D), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "permutation, causal",
    [(PermutationIdentity, torch.tril), (PermutationFlip, torch.triu)],
)
def test_block_is_autoregressive(permutation: type, causal) -> None:
    """Output token i only depends on input tokens <= i (in permuted order)."""
    block = MetaBlock(T, D, PROJ, permutation(T), num_layers=2).eval()
    fill_params(block)
    x, _ = make_inputs("uncond")
    jac = torch.autograd.functional.jacobian(lambda x_: block(x_)[0], x[:1])
    dep = jac[0, :, :, 0].abs().sum(dim=(1, 3))  # (T_out, T_in) dependency matrix
    assert (dep.diagonal() > 0).all()
    torch.testing.assert_close(dep, causal(dep))


def test_first_token_is_identity() -> None:
    """The first token (in permuted order) passes through a block unchanged."""
    x, _ = make_inputs("uncond")
    for permutation, idx in [(PermutationIdentity, 0), (PermutationFlip, -1)]:
        block = MetaBlock(T, D, PROJ, permutation(T)).eval()
        z, _ = block(x)
        torch.testing.assert_close(z[:, idx], x[:, idx])


def test_attention_implementations_agree() -> None:
    attn = Attention(PROJ, 32).eval()
    x = torch.randn(B, T, PROJ)
    mask = torch.tril(torch.ones(T, T))
    with torch.no_grad():
        torch.testing.assert_close(
            attn.forward_spda(x, mask), attn.forward_base(x, mask), atol=1e-5, rtol=1e-4
        )


def test_dropped_class_label_uses_mean_embedding() -> None:
    """y == -1 (label dropped for CFG training) is equivalent to y = None."""
    model = make_model("class")
    x, y = make_inputs("class")
    with torch.no_grad():
        z_none, _, _ = model(x, None)
        z_dropped, _, _ = model(x, torch.full_like(y, -1))
        z_cond, _, _ = model(x, y)
    torch.testing.assert_close(z_dropped, z_none)
    assert not torch.allclose(z_cond, z_none)


@pytest.mark.parametrize("variant", ["class", "continuous"])
def test_conditioning_affects_output(variant: str) -> None:
    model = make_model(variant)
    x, y = make_inputs(variant)
    with torch.no_grad():
        z, _, _ = model(x, y)
        z_flipped, _, _ = model(x, y.flip(0))
    assert not torch.allclose(z, z_flipped)


def test_cannot_use_both_conditioning_types() -> None:
    with pytest.raises(AssertionError, match="Only one of"):
        TarFlow(T, D, PROJ, 1, 1, num_classes=NUM_CLASSES, cond_dim=COND_DIM)


# ---------------------------------------------------------------- reverse


@pytest.mark.parametrize("variant", VARIANTS)
def test_reverse_inverts_forward(variant: str, use_spda: bool) -> None:
    model = make_model(variant)
    x, y = make_inputs(variant)
    with torch.no_grad():
        z, _, _ = model(x, y)
        z_orig = z.clone()
        x_recon = model.reverse(z, y)
    torch.testing.assert_close(x_recon, x, atol=1e-4, rtol=0)
    torch.testing.assert_close(z, z_orig)  # input not modified in place


def test_forward_inverts_reverse() -> None:
    """Sampling direction: z -> x -> z."""
    model = make_model("uncond")
    z = torch.randn(B, T, D)
    with torch.no_grad():
        z_recon, _, _ = model(model.reverse(z))
    torch.testing.assert_close(z_recon, z, atol=1e-4, rtol=0)


def test_reverse_return_sequence() -> None:
    model = make_model("uncond")
    z = torch.randn(B, T, D)
    with torch.no_grad():
        seq = model.reverse(z, return_sequence=True)
        x = model.reverse(z)
    assert isinstance(seq, list)
    assert len(seq) == len(model.blocks) + 1
    torch.testing.assert_close(seq[0], z)
    torch.testing.assert_close(seq[-1], x)


def test_reverse_scales_by_prior_std() -> None:
    model = make_model("vp")
    z = torch.randn(B, T, D)
    with torch.no_grad():
        x = model.reverse(z)
        model.var.fill_(4.0)
        x_scaled = model.reverse(z / 2)
    torch.testing.assert_close(x_scaled, x)


def test_reverse_resets_sample_mode() -> None:
    """KV caches must be emptied after sampling, or later forward passes break."""
    model = make_model("class")
    x, y = make_inputs("class")
    with torch.no_grad():
        model.reverse(x, y, guidance=1.0)
    for m in model.modules():
        if isinstance(m, Attention):
            assert not m.sample
            assert m.k_cache == {"cond": None, "uncond": None}
            assert m.v_cache == {"cond": None, "uncond": None}


def test_guidance() -> None:
    model = make_model("class")
    z, y = make_inputs("class")
    with torch.no_grad():
        x = model.reverse(z, y)
        x_none = model.reverse(z, y, guidance=1.0, guide_what="")
        x_guided = model.reverse(z, y, guidance=1.0)
        x_guided_a = model.reverse(z, y, guidance=1.0, guide_what="a")
        x_annealed = model.reverse(z, y, guidance=1.0, annealed_guidance=True)
    assert all(t.isfinite().all() for t in (x_guided, x_guided_a, x_annealed))
    torch.testing.assert_close(x_none, x)
    assert not torch.allclose(x_guided, x)
    assert not torch.allclose(x_guided_a, x_guided)
    assert not torch.allclose(x_annealed, x_guided)


# ---------------------------------------------------------------- training


def test_get_loss() -> None:
    model = make_model("uncond")
    z = torch.full((B, T, D), 2.0)
    logdets = torch.tensor([1.0, 3.0])
    torch.testing.assert_close(model.get_loss(z, logdets), torch.tensor(0.0))


def test_update_prior() -> None:
    model = make_model("vp")
    z = torch.full((B, T, D), 3.0)
    model.update_prior(z)
    expected = 1 + model.VAR_LR * (9.0 - 1)  # lerp from the initial var of 1
    torch.testing.assert_close(model.var, torch.full((T, D), expected))


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_parameters_receive_gradients(variant: str) -> None:
    model = make_model(variant)
    z, _, logdets = model(*make_inputs(variant))
    model.get_loss(z, logdets).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert p.grad.isfinite().all(), name
        assert p.grad.any(), name


def test_training_reduces_loss() -> None:
    torch.manual_seed(0)
    model = TarFlow(T, D, PROJ, num_blocks=2, layers_per_block=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(32, T, D) * 3 + 1
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        z, _, logdets = model(x)
        loss = model.get_loss(z, logdets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


# ---------------------------------------------------------------- checkpoints

# parameter/buffer names & shapes that existing checkpoints rely on
BLOCK_STATE = {
    "pos_embed": (T, PROJ),
    "attn_mask": (T, T),
    "proj_in.weight": (PROJ, D),
    "proj_in.bias": (PROJ,),
    "attn_blocks.0.attention.norm.weight": (PROJ,),
    "attn_blocks.0.attention.norm.bias": (PROJ,),
    "attn_blocks.0.attention.qkv.weight": (3 * PROJ, PROJ),
    "attn_blocks.0.attention.qkv.bias": (3 * PROJ,),
    "attn_blocks.0.attention.proj.weight": (PROJ, PROJ),
    "attn_blocks.0.attention.proj.bias": (PROJ,),
    "attn_blocks.0.mlp.norm.weight": (PROJ,),
    "attn_blocks.0.mlp.norm.bias": (PROJ,),
    "attn_blocks.0.mlp.main.0.weight": (4 * PROJ, PROJ),
    "attn_blocks.0.mlp.main.0.bias": (4 * PROJ,),
    "attn_blocks.0.mlp.main.2.weight": (PROJ, 4 * PROJ),
    "attn_blocks.0.mlp.main.2.bias": (PROJ,),
    "proj_out.weight": (2 * D, PROJ),
    "proj_out.bias": (2 * D,),
}
VARIANT_STATE = {
    "uncond": {},
    "vp": {"proj_out.weight": (D, PROJ), "proj_out.bias": (D,)},
    "class": {"class_embed": (NUM_CLASSES, 1, PROJ)},
    "continuous": {"y_proj.weight": (PROJ, COND_DIM), "y_proj.bias": (PROJ,)},
}


@pytest.mark.parametrize("variant", VARIANTS)
def test_state_dict_layout(variant: str) -> None:
    """Changing parameter names or shapes breaks loading of existing checkpoints."""
    model = make_model(variant, num_blocks=2, layers=1)
    block_state = {**BLOCK_STATE, **VARIANT_STATE[variant]}
    expected = {"var": (T, D)}
    for i in range(2):
        expected.update({f"blocks.{i}.{k}": v for k, v in block_state.items()})
    actual = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    assert actual == expected


def test_get_tarflow_model(tmp_path) -> None:
    config = {
        "tarflow": {
            "z_dim": T,
            "token_size": D,
            "projection_dims": PROJ,
            "num_blocks": 2,
            "layers_per_block": 1,
            "nvp": True,
            "num_classes": 0,
        }
    }
    model = get_tarflow_model(config, input_dims=None, cond_dim=COND_DIM)
    assert isinstance(model, TarFlow)
    assert (model.num_tokens, model.token_size) == (T, D)
    assert len(model.blocks) == 2
    assert all(b.continuous_y for b in model.blocks)

    fill_params(model, seed=42)
    ckpt = tmp_path / "model.pth"
    torch.save(model.state_dict(), ckpt)
    loaded = get_tarflow_model(
        config, input_dims=None, cond_dim=COND_DIM, ckpt_file=ckpt
    )
    for (name, a), (_, b) in zip(
        model.state_dict().items(), loaded.state_dict().items()
    ):
        torch.testing.assert_close(a, b, msg=name)
