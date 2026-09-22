#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import torch


def _prefix_alias(t: torch.Tensor, dim: int, length: int) -> torch.Tensor:
    """Return `t[..., :length, ...]` along `dim` as a tensor that aliases the
    same memory but owns an independent version counter.

    A normal view shares its base's version counter, so any later write to the
    buffer invalidates every prefix already saved for backward -- even when the
    write lands in a region no saved prefix covers. Aliasing the storage
    directly opts out of that check. See `_CacheWrite` for why it is safe.
    """
    v = t.narrow(dim, 0, length)
    out = torch.empty(0, dtype=t.dtype, device=t.device)
    out.set_(v.untyped_storage(), v.storage_offset(), v.shape, v.stride())
    return out


class _CacheWrite(torch.autograd.Function):
    """Append one step's k (or v) to a preallocated cache, differentiably.

    Each step writes `[pos, pos + n)` and reads back `[0, pos + n)`, so the
    region a step writes is disjoint from every prefix earlier steps already
    handed to attention. The buffer's contents are therefore still valid at
    backward time, and all T steps can share a single allocation instead of
    each materialising its own growing copy -- O(T) memory instead of O(T^2).

    `prev` is the previous step's prefix, threaded through as a real graph edge
    so that autograd accumulates each step's share of the gradient and orders
    the backward nodes itself. Backward is then just a split: the first `pos`
    positions belong to earlier steps, the last `n` to this one.
    """

    @staticmethod
    def forward(ctx, prev, x, buf, pos, seq_dim):
        n = x.size(seq_dim)
        with torch.no_grad():
            buf.narrow(seq_dim, pos, n).copy_(x)
        ctx.pos, ctx.n, ctx.seq_dim = pos, n, seq_dim
        ctx.has_prev = prev is not None
        return _prefix_alias(buf, seq_dim, pos + n)

    @staticmethod
    def backward(ctx, grad_out):
        pos, n, dim = ctx.pos, ctx.n, ctx.seq_dim
        grad_prev = grad_out.narrow(dim, 0, pos) if ctx.has_prev else None
        return grad_prev, grad_out.narrow(dim, pos, n), None, None, None


class Permutation(torch.nn.Module):
    def __init__(self, seq_length: int):
        super().__init__()
        self.seq_length = seq_length

    def forward(
        self, x: torch.Tensor, dim: int = 1, inverse: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError("Overload me")


class PermutationIdentity(Permutation):
    def forward(
        self, x: torch.Tensor, dim: int = 1, inverse: bool = False
    ) -> torch.Tensor:
        return x


class PermutationFlip(Permutation):
    def forward(
        self, x: torch.Tensor, dim: int = 1, inverse: bool = False
    ) -> torch.Tensor:
        return x.flip(dims=[dim])


class Attention(torch.nn.Module):
    USE_SPDA: bool = True

    def __init__(self, in_channels: int, head_channels: int):
        assert in_channels % head_channels == 0
        super().__init__()
        self.norm = torch.nn.LayerNorm(in_channels)
        self.qkv = torch.nn.Linear(in_channels, in_channels * 3)
        self.proj = torch.nn.Linear(in_channels, in_channels)
        self.num_heads = in_channels // head_channels
        self.sqrt_scale = head_channels ** (-0.25)
        self.sample = False
        self.cache_max_len = 0
        self.k_cache: dict[str, torch.Tensor | None] = {"cond": None, "uncond": None}
        self.v_cache: dict[str, torch.Tensor | None] = {"cond": None, "uncond": None}
        self.k_prefix: dict[str, torch.Tensor | None] = {"cond": None, "uncond": None}
        self.v_prefix: dict[str, torch.Tensor | None] = {"cond": None, "uncond": None}
        self.cache_len: dict[str, int] = {"cond": 0, "uncond": 0}

    def _write_cache(
        self, which_cache: str, k: torch.Tensor, v: torch.Tensor, seq_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write k/v for the current step into a preallocated buffer along
        `seq_dim` and return the valid (in-use) prefix of that buffer.

        The buffer is allocated lazily on the first call (once we know the
        batch size, dtype, device and per-token shape) with a fixed length of
        `self.cache_max_len`, and filled in place step by step. The write goes
        through `_CacheWrite` so that gradients flow back through the cache
        without the buffer being copied per step.
        """
        pos = self.cache_len[which_cache]
        if self.k_cache[which_cache] is None:
            shape = list(k.shape)
            shape[seq_dim] = self.cache_max_len
            self.k_cache[which_cache] = k.new_zeros(shape)
            self.v_cache[which_cache] = v.new_zeros(shape)
        k_prefix = _CacheWrite.apply(
            self.k_prefix[which_cache], k, self.k_cache[which_cache], pos, seq_dim
        )
        v_prefix = _CacheWrite.apply(
            self.v_prefix[which_cache], v, self.v_cache[which_cache], pos, seq_dim
        )
        self.k_prefix[which_cache] = k_prefix
        self.v_prefix[which_cache] = v_prefix
        self.cache_len[which_cache] = pos + k.size(seq_dim)
        return k_prefix, v_prefix

    def forward_spda(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        temp: float = 1.0,
        which_cache: str = "cond",
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = (
            self.qkv(x)
            .reshape(B, T, 3 * self.num_heads, -1)
            .transpose(1, 2)
            .chunk(3, dim=1)
        )  # (b, h, t, d)

        if self.sample:
            # note that sequence dimension is now 2
            k, v = self._write_cache(which_cache, k, v, seq_dim=2)

        scale = self.sqrt_scale**2 / temp
        if mask is not None:
            mask = mask.bool()
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale
        )
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward_base(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        temp: float = 1.0,
        which_cache: str = "cond",
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).chunk(3, dim=2)
        if self.sample:
            k, v = self._write_cache(which_cache, k, v, seq_dim=1)

        attn = (
            torch.einsum("bmhd,bnhd->bmnh", q * self.sqrt_scale, k * self.sqrt_scale)
            / temp
        )
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        attn = attn.float().softmax(dim=-2).type(attn.dtype)
        x = torch.einsum("bmnh,bnhd->bmhd", attn, v)
        x = x.reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        temp: float = 1.0,
        which_cache: str = "cond",
    ) -> torch.Tensor:
        if self.USE_SPDA:
            return self.forward_spda(x, mask, temp, which_cache)
        return self.forward_base(x, mask, temp, which_cache)


class MLP(torch.nn.Module):
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(channels)
        self.main = torch.nn.Sequential(
            torch.nn.Linear(channels, channels * expansion),
            torch.nn.GELU(),
            torch.nn.Linear(channels * expansion, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(self.norm(x.float()).type(x.dtype))


class AttentionBlock(torch.nn.Module):
    def __init__(self, channels: int, head_channels: int, expansion: int = 4):
        super().__init__()
        self.attention = Attention(channels, head_channels)
        self.mlp = MLP(channels, expansion)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_temp: float = 1.0,
        which_cache: str = "cond",
    ) -> torch.Tensor:
        x = x + self.attention(x, attn_mask, attn_temp, which_cache)
        x = x + self.mlp(x)
        return x


class MetaBlock(torch.nn.Module):
    attn_mask: torch.Tensor

    def __init__(
        self,
        num_tokens: int,
        token_size: int,
        projection_dims: int,
        permutation: Permutation,
        num_layers: int = 1,
        head_dim: int = 64,
        expansion: int = 4,
        nvp: bool = True,
        num_classes: int = 0,
        cond_dim: int = 0,
    ):
        super().__init__()

        assert num_classes == 0 or cond_dim == 0, (
            "Only one of num_classes or cond_dim can be non-zero."
        )

        self.can_have_y = num_classes > 0 or cond_dim > 0
        self.continuous_y = cond_dim > 0

        self.proj_in = torch.nn.Linear(token_size, projection_dims)
        self.pos_embed = torch.nn.Parameter(
            torch.randn(num_tokens, projection_dims) * 1e-2
        )

        if num_classes:
            self.class_embed = torch.nn.Parameter(
                torch.randn(num_classes, 1, projection_dims) * 1e-2
            )
        else:
            self.class_embed = None

        if self.continuous_y:
            self.y_proj = torch.nn.Linear(cond_dim, projection_dims)

        self.attn_blocks = torch.nn.ModuleList(
            [
                AttentionBlock(projection_dims, head_dim, expansion)
                for _ in range(num_layers)
            ]
        )
        self.nvp = nvp
        output_dim = token_size * 2 if nvp else token_size
        self.proj_out = torch.nn.Linear(projection_dims, output_dim)
        # self.proj_out.weight.data.fill_(0.0)
        self.permutation = permutation
        self.register_buffer(
            "attn_mask", torch.tril(torch.ones(num_tokens, num_tokens))
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        x_in = x
        x = self.proj_in(x) + pos_embed

        if self.can_have_y:
            if self.continuous_y:
                # y           : (batch_size, cond_dim)
                # y_proj(y)   : (batch_size, projection_dims)
                # y_embedding : (batch_size, 1, projection_dims)
                # x           : (batch_size, num_tokens, projection_dims)
                y_embedding = self.y_proj(y).unsqueeze(1)
                x = x + y_embedding
            else:
                if y is not None:
                    if (y < 0).any():
                        m = (y < 0).float().view(-1, 1, 1)
                        class_embed = (1 - m) * self.class_embed[
                            y
                        ] + m * self.class_embed.mean(dim=0)
                    else:
                        class_embed = self.class_embed[y]
                    x = x + class_embed
                else:
                    x = x + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            x = block(x, self.attn_mask)
        x = self.proj_out(x)
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        scale = (-xa.float()).exp().type(xa.dtype)
        return self.permutation((x_in - xb) * scale, inverse=True), -xa.mean(dim=[1, 2])

    def reverse_step(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        i: int,
        y: torch.Tensor | None = None,
        attn_temp: float = 1.0,
        which_cache: str = "cond",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x   : (batch_size, num_tokens, token_size)
        x_in = x[:, i : i + 1]  # get i-th token but keep the token size dimension
        # x_in: (batch_size, 1, token_size)
        x = self.proj_in(x_in) + pos_embed[i : i + 1]
        # x   : (batch_size, 1, projection_dims)

        if self.can_have_y:
            if self.continuous_y:
                # y           : (batch_size, cond_dim)
                # y_proj(y)   : (batch_size, projection_dims)
                # y_embedding : (batch_size, 1, projection_dims)
                # x           : (batch_size, 1, projection_dims)
                y_embedding = self.y_proj(y).unsqueeze(1)
                x = x + y_embedding
            else:
                if y is not None:
                    x = x + self.class_embed[y]
                else:
                    x = x + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            x = block(
                x, attn_temp=attn_temp, which_cache=which_cache
            )  # here we use kv caching, so no attn_mask
        x = self.proj_out(x)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        return xa, xb

    def set_sample_mode(self, flag: bool = True, max_len: int = 0):
        for m in self.modules():
            if isinstance(m, Attention):
                m.sample = flag
                m.cache_max_len = max_len
                m.k_cache = {"cond": None, "uncond": None}
                m.v_cache = {"cond": None, "uncond": None}
                m.cache_len = {"cond": 0, "uncond": 0}

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = "ab",
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
    ) -> torch.Tensor:
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        T = x.size(1)
        self.set_sample_mode(True, T - 1)
        for i in range(x.size(1) - 1):
            za, zb = self.reverse_step(x, pos_embed, i, y, which_cache="cond")
            if guidance > 0 and guide_what:
                za_u, zb_u = self.reverse_step(
                    x, pos_embed, i, None, attn_temp=attn_temp, which_cache="uncond"
                )
                if annealed_guidance:
                    g = (i + 1) / (T - 1) * guidance
                else:
                    g = guidance
                if "a" in guide_what:
                    za = za + g * (za - za_u)
                if "b" in guide_what:
                    zb = zb + g * (zb - zb_u)

            scale = (
                za[:, 0].float().exp().type(za.dtype)
            )  # get rid of the sequence dimension
            x[:, i + 1] = x[:, i + 1] * scale + zb[:, 0]
        self.set_sample_mode(False)
        return self.permutation(x, inverse=True)


class Model(torch.nn.Module):
    VAR_LR: float = 0.1
    var: torch.Tensor

    def __init__(
        self,
        num_tokens: int,
        token_size: int,
        projection_dims: int,
        num_blocks: int,
        layers_per_block: int,
        nvp: bool = True,
        num_classes: int = 0,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_size = token_size
        permutations = [
            PermutationIdentity(num_tokens),
            PermutationFlip(num_tokens),
        ]

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MetaBlock(
                    num_tokens,
                    token_size,
                    projection_dims,
                    permutations[i % 2],
                    layers_per_block,
                    nvp=nvp,
                    num_classes=num_classes,
                    cond_dim=cond_dim,
                )
            )
        self.blocks = torch.nn.ModuleList(blocks)
        # prior for nvp mode should be all ones, but needs to be learnd for the vp mode
        self.register_buffer("var", torch.ones(num_tokens, token_size))

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        outputs = []
        logdets = torch.zeros((), device=x.device)
        for block in self.blocks:
            x, logdet = block(x, y)
            logdets = logdets + logdet
            outputs.append(x)
        return x, outputs, logdets

    def update_prior(self, z: torch.Tensor):
        z2 = (z**2).mean(dim=0)
        self.var.lerp_(z2.detach(), weight=self.VAR_LR)

    def get_loss(self, z: torch.Tensor, logdets: torch.Tensor):
        return 0.5 * z.pow(2).mean() - logdets.mean()

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = "ab",
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
        return_sequence: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        seq = [x]
        x = x * self.var.sqrt()
        for block in reversed(self.blocks):
            x = block.reverse(x, y, guidance, guide_what, attn_temp, annealed_guidance)
            seq.append(x)

        if not return_sequence:
            return x
        else:
            return seq

def get_tarflow_model(config, input_dims, cond_dim=0, ckpt_file=None):
    tarflow_model = Model(
        num_tokens=config["tarflow"]["z_dim"],
        token_size=config["tarflow"]["token_size"],
        projection_dims=config["tarflow"]["projection_dims"],
        num_blocks=config["tarflow"]["num_blocks"],
        layers_per_block=config["tarflow"]["layers_per_block"],
        nvp=config["tarflow"]["nvp"],
        num_classes=config["tarflow"]["num_classes"],
        cond_dim=cond_dim,
    )

    if ckpt_file:
        tarflow_model.load_state_dict(torch.load(ckpt_file))
        print(f"loaded weights from {ckpt_file}")

    return tarflow_model
