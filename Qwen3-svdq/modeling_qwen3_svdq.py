# This file is derived from `modeling_qwen3.py` and adapted for SVDQ (SVD-Quant) inference.
# NOTE: In upstream Transformers, modeling files may be auto-generated. In this project we maintain
# a separate `modeling_qwen3_svdq.py` for the custom backend.

from collections.abc import Callable
from typing import Optional

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub, use_kernel_func_from_hub, use_kernelized_func
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import maybe_autocast, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs

# SVDQ config (project local)
from .configuration_qwen3_svdq import Qwen3SvdqConfig


# ============================================================
# Optional CUDA path (stub hook points for later kernel integration)
# ============================================================

try:
    # You will provide this module later (torch.ops bindings or python wrapper).
    # Expected interface (to be implemented later):
    #   linear_nvfp4(x, core0, core1, w_u8, w_scale_fp8, w_scale2_f32, bias=None, lora=None, smooth=None) -> y
    #   linear_fp6 (x, core0, core1, w_i8, quant_scale, quant_dim, bias=None, lora=None, smooth=None) -> y
    from . import svdq_kernels  # type: ignore
except Exception:  # pragma: no cover
    svdq_kernels = None


# ============================================================
# SVDQ Linear building block
# ============================================================

_FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

def _unpack_fp4_nibbles(packed_u8: torch.Tensor, in_features: int) -> torch.Tensor:
    """Unpack uint8 packed nibbles -> uint8 codes [out, in_features] with codes in [0..15]."""
    # packed_u8: [out, in/2]
    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    codes = torch.stack([low, high], dim=-1).reshape(packed_u8.shape[0], -1)
    return codes[:, :in_features].contiguous()

def _dequant_nvfp4(
    w_u8: torch.Tensor,
    w_scale: torch.Tensor,
    w_scale2: torch.Tensor,
    in_features: int,
    group_size: int = 16,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize NVFP4-like packed weights produced by our v1 packer:
      - w_u8: uint8 [out, in/2]  (two 4-bit codes per byte)
      - w_scale: fp8_e4m3fn (or fp16 fallback) [out, in/16]
      - w_scale2: fp32 scalar
    Returns dense bf16/fp16 weight [out, in].
    """
    out = w_u8.shape[0]
    codes = _unpack_fp4_nibbles(w_u8, in_features)  # [out,in]
    sign = ((codes >> 3) & 0x01).to(torch.float32)  # 0/1
    mag = (codes & 0x07).to(torch.long)             # 0..7
    vals = _FP4_LEVELS.to(codes.device)[mag]        # [out,in] float32 magnitudes
    vals = torch.where(sign > 0, -vals, vals)

    if in_features % group_size != 0:
        raise ValueError(f"NVFP4 dequant expects in_features % {group_size} == 0, got {in_features}")
    nblk = in_features // group_size
    vals = vals.view(out, nblk, group_size)

    scale = w_scale.to(torch.float32).view(out, nblk, 1)
    scale2 = w_scale2.to(torch.float32).view(1, 1, 1)
    w = (vals * scale * scale2).view(out, in_features)
    return w.to(out_dtype)

def _dequant_fp6(
    w_i8: torch.Tensor,
    quant_scale: torch.Tensor,
    quant_dim: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize our FP6 placeholder (int6 grid stored in int8). Returns dense [out,in]."""
    w = w_i8.to(torch.float32)
    s = quant_scale.to(torch.float32)
    if quant_dim == 0:
        w = w * s.view(1, -1)
    elif quant_dim == 1:
        w = w * s.view(-1, 1)
    else:
        raise ValueError(f"quant_dim must be 0 or 1, got {quant_dim}")
    return w.to(out_dtype)


class SVDQLinear(nn.Module):
    """
    SVDQ Linear (forward-only):

      y = (x @ core1^T) @ core0^T  +  x @ dequant(residual)^T  +  LoRA(x)  + bias

    Rank is NOT stored in config; it is inferred from checkpoint tensor shapes. Therefore, core0/core1
    (and optional residual / LoRA tensors) are created lazily during `_load_from_state_dict`.

    TP policy note:
      The choice of TP mode (colwise/rowwise) should be decided at loader time, but the per-tensor sharding logic
      belongs in this module since core0/core1 (and LoRA) shard differently depending on mode.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, *, group_size: int = 16):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)

        # created lazily at load time
        self.core0: Optional[nn.Parameter] = None
        self.core1: Optional[nn.Parameter] = None

        self.bias: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16), requires_grad=False) if bias else None
        )

        # residual (mutually exclusive)
        self.register_buffer("nvfp4_w", None, persistent=True)
        self.register_buffer("nvfp4_scale", None, persistent=True)
        self.register_buffer("nvfp4_scale2", None, persistent=True)

        self.register_buffer("fp6_w", None, persistent=True)
        self.register_buffer("quant_scale", None, persistent=True)
        self.register_buffer("quant_dim", None, persistent=True)

        # optional smooth
        self.register_buffer("smooth_factor", None, persistent=True)

        # optional LoRA
        self.lora_A: Optional[nn.Parameter] = None
        self.lora_B: Optional[nn.Parameter] = None
        self.register_buffer("lora_alpha", None, persistent=True)

        # execution mode
        self.use_kernel = False  # loader can switch it on

    def _ensure_param(self, name: str, t: torch.Tensor) -> None:
        if getattr(self, name, None) is None:
            setattr(self, name, nn.Parameter(torch.empty_like(t), requires_grad=False))

    def _ensure_buffer(self, name: str, t: torch.Tensor) -> None:
        if getattr(self, name, None) is None:
            self.register_buffer(name, torch.empty_like(t), persistent=True)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # core factors
        for kn in ("core0", "core1"):
            kk = prefix + kn
            if kk in state_dict:
                self._ensure_param(kn, state_dict[kk])

        # bias
        kb = prefix + "bias"
        if kb in state_dict and self.bias is None:
            self.bias = nn.Parameter(torch.empty_like(state_dict[kb]), requires_grad=False)

        # nvfp4 residual
        for kn in ("nvfp4_w", "nvfp4_scale", "nvfp4_scale2"):
            kk = prefix + kn
            if kk in state_dict:
                self._ensure_buffer(kn, state_dict[kk])

        # fp6 residual
        for kn in ("fp6_w", "quant_scale", "quant_dim"):
            kk = prefix + kn
            if kk in state_dict:
                self._ensure_buffer(kn, state_dict[kk])

        # smooth
        ks = prefix + "smooth_factor"
        if ks in state_dict:
            self._ensure_buffer("smooth_factor", state_dict[ks])

        # lora
        kA = prefix + "lora_A"
        kB = prefix + "lora_B"
        kAlpha = prefix + "lora_alpha"
        if kA in state_dict and self.lora_A is None:
            self.lora_A = nn.Parameter(torch.empty_like(state_dict[kA]), requires_grad=False)
        if kB in state_dict and self.lora_B is None:
            self.lora_B = nn.Parameter(torch.empty_like(state_dict[kB]), requires_grad=False)
        if kAlpha in state_dict:
            self._ensure_buffer("lora_alpha", state_dict[kAlpha])

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    # ---- forward paths ----
    def _forward_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        if self.core0 is None or self.core1 is None:
            raise RuntimeError("SVDQLinear called before core0/core1 are loaded.")

        # placeholder smooth semantics (can be updated later)
        if self.smooth_factor is not None:
            x = x * self.smooth_factor.to(dtype=x.dtype, device=x.device)

        x_bf16 = x.to(torch.bfloat16)

        z = torch.matmul(x_bf16, self.core1.to(dtype=torch.bfloat16).t())
        y = torch.matmul(z, self.core0.to(dtype=torch.bfloat16).t())

        if self.nvfp4_w is not None:
            w_r = _dequant_nvfp4(
                self.nvfp4_w.to(device=x.device),
                self.nvfp4_scale.to(device=x.device),
                self.nvfp4_scale2.to(device=x.device),
                in_features=self.in_features,
                group_size=self.group_size,
                out_dtype=torch.bfloat16,
            )
            y = y + torch.matmul(x_bf16, w_r.t())
        elif self.fp6_w is not None:
            qd = int(self.quant_dim.item()) if self.quant_dim is not None else 0
            w_r = _dequant_fp6(
                self.fp6_w.to(device=x.device),
                self.quant_scale.to(device=x.device),
                quant_dim=qd,
                out_dtype=torch.bfloat16,
            )
            y = y + torch.matmul(x_bf16, w_r.t())

        if self.lora_A is not None and self.lora_B is not None:
            alpha = float(self.lora_alpha.item()) if self.lora_alpha is not None else float(self.lora_A.shape[0])
            r = float(self.lora_A.shape[0])
            scale = alpha / max(r, 1.0)
            tmp = torch.matmul(x_bf16, self.lora_A.to(dtype=torch.bfloat16).t())
            y = y + scale * torch.matmul(tmp, self.lora_B.to(dtype=torch.bfloat16).t())

        if self.bias is not None:
            y = y + self.bias.to(dtype=y.dtype, device=y.device)

        return y

    def _forward_kernel(self, x: torch.Tensor) -> torch.Tensor:
        if svdq_kernels is None:
            return self._forward_pytorch(x)

        if self.nvfp4_w is not None:
            return svdq_kernels.linear_nvfp4(  # type: ignore[attr-defined]
                x, self.core0, self.core1,
                self.nvfp4_w, self.nvfp4_scale, self.nvfp4_scale2,
                self.bias, self.smooth_factor,
                self.lora_A, self.lora_B, self.lora_alpha,
            )
        if self.fp6_w is not None:
            return svdq_kernels.linear_fp6(  # type: ignore[attr-defined]
                x, self.core0, self.core1,
                self.fp6_w, self.quant_scale, self.quant_dim,
                self.bias, self.smooth_factor,
                self.lora_A, self.lora_B, self.lora_alpha,
            )
        return self._forward_pytorch(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_kernel(x) if self.use_kernel else self._forward_pytorch(x)


# ============================================================
# Qwen3 components: RMSNorm, RoPE, Attention, MLP (SVDQ)
# ============================================================

@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3SvdqConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = SVDQLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = SVDQLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = SVDQLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3SvdqConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: Qwen3SvdqConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


@use_kernelized_func(apply_rotary_pos_emb)
class Qwen3Attention(nn.Module):
    """Multi-headed attention adapted to SVDQLinear projections."""

    def __init__(self, config: Qwen3SvdqConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = SVDQLinear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = SVDQLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = SVDQLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = SVDQLinear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3SvdqConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config, layer_idx=layer_idx)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Qwen3SvdqPreTrainedModel(PreTrainedModel):
    config: Qwen3SvdqConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }


@auto_docstring
class Qwen3SvdqModel(Qwen3SvdqPreTrainedModel):
    def __init__(self, config: Qwen3SvdqConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class Qwen3SvdqForCausalLM(Qwen3SvdqPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: Qwen3SvdqConfig):
        super().__init__(config)
        self.model = Qwen3SvdqModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen3SvdqForSequenceClassification(GenericForSequenceClassification, Qwen3SvdqPreTrainedModel):
    pass


class Qwen3SvdqForTokenClassification(GenericForTokenClassification, Qwen3SvdqPreTrainedModel):
    pass


class Qwen3SvdqForQuestionAnswering(GenericForQuestionAnswering, Qwen3SvdqPreTrainedModel):
    base_model_prefix = "transformer"


__all__ = [
    "Qwen3SvdqForCausalLM",
    "Qwen3SvdqForQuestionAnswering",
    "Qwen3SvdqPreTrainedModel",
    "Qwen3SvdqModel",
    "Qwen3SvdqForSequenceClassification",
    "Qwen3SvdqForTokenClassification",
    "SVDQLinear",
]
