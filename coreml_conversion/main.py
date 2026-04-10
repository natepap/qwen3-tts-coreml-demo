"""
Qwen3-TTS → CoreML Conversion Pipeline
=======================================

WHY NOT EXPORT THE FULL MODEL
------------------------------
Three things in Qwen3TTSTalkerForConditionalGeneration.forward() make it
un-exportable as-is:

  1. DATA-DEPENDENT BRANCHING  (line ~1665)
       if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
     torch.export / jit.trace cannot follow branches that depend on
     runtime tensor shapes.

  2. PYTHON OBJECT INSIDE FORWARD  (line ~1671)
       predictor_result = self.code_predictor.generate(...)
     Calling .generate() (which has its own sampling loop) inside forward()
     is not a tensor operation and cannot be traced.

  3. VMAP IN CREATE_CAUSAL_MASK  (line ~1510 in Qwen3TTSTalkerModel)
       causal_mask = create_causal_mask(config=…, …)
     transformers.masking_utils.create_causal_mask() dispatches to
     sdpa_mask_older_torch() which calls _vmap_for_bhqkv().  During
     torch.export this raises:
       RuntimeError: Attempting to use FunctionalTensor on its own.
                     Instead, please use it with FunctionalTensorMode()

THE FIX: CALL LAYERS DIRECTLY
-------------------------------
Instead of exporting the high-level class, we export thin nn.Module wrappers
that call the underlying decoder layers directly, passing a **pre-computed
4-D float attention mask** (shape 1, 1, q_len, kv_len).  This:

  • Bypasses create_causal_mask entirely → no vmap, no FunctionalTensor error
  • Removes all Python-level control flow from the traced graph
  • Gives coremltools a clean, statically-shaped computation graph

EXPORTED COMPONENTS
--------------------
  1. SpeakerEncoder      mel (1,T,128) → embedding (1,1024)
     Simple ECAPA-TDNN CNN stack; no attention issues.  torch.jit.trace.

  2. TalkerContextPass   embeds (1,S,1024) → logits (1,S,vocab)
     Full-context attention pass (no KV cache).  torch.jit.trace.
     Used for both prefill and, in Phase 1, decode (full recompute each step).

GENERATION LOOP (NOT EXPORTED)
--------------------------------
Sampling, EOS detection, codec_predictor.generate(), and the main
while-loop all stay in Python (and later in Swift).  CoreML only handles
the heavy tensor operations.

PHASE 2 (TODO)
---------------
  • Stateful KV-cache decode model using ct.StateType (requires iOS 18+)
    This converts decode from O(n²) to O(n) attention cost.
  • Separate CodePredictor wrapper (small 5-layer transformer).
  • SpeechDecoder wrapper (speech tokenizer's vocoder).

RECOMMENDED PACKAGE VERSIONS
------------------------------
The primary issue is architectural, not version-related.  Current versions
(torch 2.7.0, coremltools 9.0, transformers 4.57.3) work with the wrappers
below.  If you still hit BlobWriter errors, the most battle-tested fallback
stack for CoreML LLM conversion is:

  torch==2.5.1
  torchaudio==2.5.1
  coremltools==8.1.0        # well-documented; iOS 18 stateful cache support
  transformers==4.57.3      # keep — model code depends on its newer APIs

DO NOT downgrade transformers: the model imports masking_utils,
modeling_rope_utils, can_return_tuple, etc. that don't exist in older versions.

USAGE
------
  python -m coreml_conversion.main \\
      --model_path /path/to/Qwen3-TTS-Base \\
      --output_dir ./coreml_models \\
      --seq_len 256 \\
      --target ios17
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct


# ──────────────────────────────────────────────────────────────────────────────
# Monkey-patch: CoreML-safe mRoPE  (must be applied before tracing)
#
# TWO BUGS in the original apply_multimodal_rotary_pos_emb:
#
#  Bug 1 – non-interleaved path (line ~714 in model file):
#       cos.split(mrope_section * 2, dim=-1)
#   Generates aten::split_with_sizes with a 1-D int tensor as the sizes arg.
#   coremltools calls int(np_array_of_shape_6) → "only 0-dimensional arrays
#   can be converted to Python scalars".
#   Fix: explicit Python-int slices  cos[mod, :, :, start:end].
#
#  Bug 2 – interleaved path (line ~702):
#       x_t[..., beg:end:step] = x[i, ..., beg:end:step]   (in-place + stride>1)
#   In-place assignment on a traced tensor is unsupported in CoreML.
#   Stride > 1 slicing (step=modality_num=3) may also be unsupported.
#   Fix: replace with torch.where using pre-computed constant boolean masks.
# ──────────────────────────────────────────────────────────────────────────────

def _coreml_apply_multimodal_rotary_pos_emb(
    q, k, cos, sin, mrope_section, mrope_interleaved=False, unsqueeze_dim=1
):
    """
    Drop-in replacement for apply_multimodal_rotary_pos_emb that is safe
    for coremltools conversion.  Handles both interleaved and non-interleaved
    mRoPE variants.

    cos / sin shape: (3, batch, seq_len, head_dim)
      dim-0 = modality: 0=temporal, 1=height, 2=width
    """

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2      # Python int in the trace (static shape)
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    if mrope_interleaved:
        # ── interleaved branch ────────────────────────────────────────────────
        # The original assigns each modality to every 3rd frequency position:
        #   temporal  → positions 0, 3, 6, …
        #   height    → positions 1, 4, 7, …  up to mrope_section[1]*3
        #   width     → positions 2, 5, 8, …  up to mrope_section[2]*3
        #
        # We reproduce this with torch.where + constant bool masks.
        # All mask values are Python bools computed from mrope_section (a Python
        # list from the model config), so they become constant tensors in the trace.

        half_dim: int  = sum(mrope_section)          # = head_dim // 2
        mod_num:  int  = len(mrope_section)          # = 3

        height_set = set(range(1, mrope_section[1] * mod_num, mod_num))
        width_set  = set(range(2, mrope_section[2] * mod_num, mod_num))

        # torch.tensor(Python list) → constant bool tensor node in the trace
        is_height = torch.tensor(
            [i in height_set for i in range(half_dim)], dtype=torch.bool
        ).view(1, 1, half_dim)    # broadcast over (batch, seq_len, half_dim)

        is_width = torch.tensor(
            [i in width_set for i in range(half_dim)], dtype=torch.bool
        ).view(1, 1, half_dim)

        def _interleave(x_in: torch.Tensor) -> torch.Tensor:
            x_h = x_in[..., :half_dim]    # (3, batch, seq_len, half_dim)
            # torch.where → CoreML mb.select — no in-place ops, no strided slices
            mixed = torch.where(is_height, x_h[1],
                                torch.where(is_width, x_h[2], x_h[0]))
            return torch.cat([mixed, mixed], dim=-1)   # (batch, seq_len, head_dim)

        cos_r = _interleave(cos).unsqueeze(unsqueeze_dim)
        sin_r = _interleave(sin).unsqueeze(unsqueeze_dim)

    else:
        # ── non-interleaved branch ────────────────────────────────────────────
        # sections = [a,b,c,a,b,c] covering the full head_dim
        sections: list = list(mrope_section) * 2

        starts: list = []
        cursor: int  = 0
        for size in sections:
            starts.append(cursor)
            cursor += size

        # cos[mod, :, :, start:end] → (batch, seq_len, size)
        # Every bound is a scalar Python int — coremltools sees only constants.
        cos_parts: list = []
        sin_parts: list = []
        for i, (start, size) in enumerate(zip(starts, sections)):
            mod = i % 3
            end = start + size
            cos_parts.append(cos[mod, :, :, start:end])
            sin_parts.append(sin[mod, :, :, start:end])

        cos_r = torch.cat(cos_parts, dim=-1).unsqueeze(unsqueeze_dim)
        sin_r = torch.cat(sin_parts, dim=-1).unsqueeze(unsqueeze_dim)

    q_embed = q * cos_r + _rotate_half(q) * sin_r
    k_embed = k * cos_r + _rotate_half(k) * sin_r
    return q_embed, k_embed


def patch_mrope_for_coreml() -> None:
    """
    Replace the module-level apply_multimodal_rotary_pos_emb in the model
    file with the CoreML-safe version above.

    Must be called BEFORE torch.jit.trace; the patch has no effect on the
    original PyTorch inference path (only the function reference is replaced).
    """
    import qwen_tts.core.models.modeling_qwen3_tts as _model_module
    _model_module.apply_multimodal_rotary_pos_emb = _coreml_apply_multimodal_rotary_pos_emb
    print("  ✓ patched apply_multimodal_rotary_pos_emb for CoreML (static indexing + no in-place ops)")


# ──────────────────────────────────────────────────────────────────────────────
# Additional patches for remaining aten::Int failures
#
# aten::Int is the TorchScript op for converting a (supposedly 0-D) tensor to
# a Python int.  coremltools tries  int(x.val)  where x.val is a numpy array;
# if x.val is NOT 0-D, the conversion fails with the same error.
#
# Two more sources remain after the mRoPE fix:
#
#  Source A – repeat_kv (eager_attention_forward calls this)
#     batch, kv_heads, slen, hd = hidden_states.shape
#     hidden_states.reshape(batch, kv_heads * n_rep, slen, hd)
#   aten::size() returns a List[int] stored in the IR; element extraction can
#   produce aten::Int on an intermediate 1-D constant.
#   Fix: avoid shape unpacking entirely using flatten(1, 2).
#
#  Source B – Qwen3TTSRMSNorm.forward
#     input_dtype = hidden_states.dtype        ← prim::dtype → int node
#     hidden_states.to(torch.float32)
#     return … hidden_states.to(input_dtype)   ← aten::to with dynamic dtype int
#   prim::dtype returns a dtype code as an int; in some coremltools versions
#   this becomes a 1-D constant that aten::Int cannot lower to a scalar.
#   Fix: use explicit literal dtypes so no prim::dtype node is emitted.
# ──────────────────────────────────────────────────────────────────────────────

def _coreml_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    CoreML-safe drop-in for repeat_kv.
    Uses flatten(1, 2) instead of shape-unpacking + reshape to avoid the
    aten::size → aten::__getitem__ → aten::Int chain that coremltools fails on.

    Equivalent output: (B, kv_heads*n_rep, S, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    # (B, kv, 1, S, hd) → expand → (B, kv, n_rep, S, hd) → flatten 1&2
    return (
        hidden_states.unsqueeze(2)
        .expand(-1, -1, n_rep, -1, -1)
        .flatten(1, 2)
    )


def _coreml_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    CoreML-safe RMSNorm that avoids prim::dtype (dynamic dtype lookup).
    Uses explicit torch.float32 literals instead of hidden_states.dtype so
    no aten::Int op is emitted for the dtype code.

    Always returns float32; downstream coremltools quantisation to float16
    is handled at conversion time via compute_precision=ct.precision.FLOAT16.
    """
    # Explicit literal dtype → prim::Constant[dtype=float32], not prim::dtype
    hidden_float = hidden_states.to(torch.float32)
    variance = hidden_float.pow(2).mean(-1, keepdim=True)
    hidden_norm = hidden_float * torch.rsqrt(variance + self.variance_epsilon)
    # Cast weight to float32 too to avoid mixed-precision mul
    return self.weight.to(torch.float32) * hidden_norm


def patch_repeat_kv_and_rmsnorm_for_coreml() -> None:
    """
    Replace repeat_kv and Qwen3TTSRMSNorm.forward with CoreML-safe versions.
    Must be called BEFORE torch.jit.trace.
    """
    import qwen_tts.core.models.modeling_qwen3_tts as _m
    _m.repeat_kv = _coreml_repeat_kv
    _m.Qwen3TTSRMSNorm.forward = _coreml_rmsnorm_forward
    print("  ✓ patched repeat_kv           (flatten instead of shape-unpack + reshape)")
    print("  ✓ patched Qwen3TTSRMSNorm     (literal float32 dtype, no prim::dtype node)")


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic helper: scan the TorchScript graph for aten::Int ops
#
# Run this BEFORE ct.convert to identify remaining int-cast failures before
# they crash the converter.  Each printed line shows the node name and what
# value it receives, helping pinpoint the exact source.
# ──────────────────────────────────────────────────────────────────────────────

def print_int_ops(traced_module: torch.jit.ScriptModule) -> int:
    """
    Inline the traced graph and print every aten::Int node found.
    Returns the count of aten::Int nodes (0 = clean, should convert fine).

    Usage:
        traced = torch.jit.trace(wrapper, example_inputs)
        n = print_int_ops(traced)
        if n == 0:
            mlmodel = ct.convert(traced, ...)
    """
    import torch._C

    graph = traced_module.graph.copy()
    torch._C._jit_pass_inline(graph)          # flatten all sub-calls into one graph
    torch._C._jit_pass_constant_propagation(graph)  # fold constants first

    nodes = list(graph.nodes())
    int_nodes = [n for n in nodes if n.kind() == "aten::Int"]

    print(f"\n  [debug] TorchScript graph has {len(nodes)} nodes total.")
    print(f"  [debug] Found {len(int_nodes)} aten::Int node(s):")

    for node in int_nodes:
        inp = list(node.inputs())
        inp_strs = []
        for i in inp:
            try:
                val = i.toIValue()
                inp_strs.append(f"{i.type()} = {val!r}")
            except Exception:
                inp_strs.append(str(i.type()))
        print(f"    {node.sourceRange()!s:60s}  inputs: [{', '.join(inp_strs)}]")

    if not int_nodes:
        print("    (none — graph should be clean for coremltools)")
    print()
    return len(int_nodes)


# ──────────────────────────────────────────────────────────────────────────────
# Shape constants — verified against defaults in configuration_qwen3_tts.py
# Update these if your checkpoint uses different sizes.
# ──────────────────────────────────────────────────────────────────────────────
NUM_TALKER_LAYERS: int = 28
KV_HEADS:          int = 8
HEAD_DIM:          int = 128
HIDDEN_DIM:        int = 1024          # talker hidden size
VOCAB_SIZE:        int = 3072          # talker codec vocab size
MEL_DIM:           int = 128
SPEAKER_DIM:       int = 1024

# Static sequence length used when tracing the Talker.
# Must be >= your longest expected prompt + any reference audio tokens.
# Larger = more memory; too small = silent truncation at inference time.
DEFAULT_SEQ_LEN: int = 256


# ──────────────────────────────────────────────────────────────────────────────
# Utility: build a 4-D additive causal mask WITHOUT vmap
# ──────────────────────────────────────────────────────────────────────────────

def make_causal_mask(q_len: int, kv_len: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Return a (1, 1, q_len, kv_len) additive causal mask.

    Attentable positions → 0.0
    Masked positions     → -inf

    This replicates what transformers.masking_utils.create_causal_mask produces
    but uses plain torch ops with no vmap, making it safe to call during tracing.
    """
    # q_idx[:, None] <= kv_idx[None, :] + offset → attend to past + self
    q_idx  = torch.arange(q_len,  dtype=torch.long).unsqueeze(1)   # (q, 1)
    kv_idx = torch.arange(kv_len, dtype=torch.long).unsqueeze(0)   # (1, kv)
    offset = kv_len - q_len                                         # for cross-attention / generation
    attend = kv_idx <= q_idx + offset                               # (q, kv) bool
    mask   = torch.where(attend,
                          torch.zeros(1, dtype=dtype),
                          torch.full((1,), float("-inf"), dtype=dtype))
    return mask.view(1, 1, q_len, kv_len)


def make_3d_position_ids(seq_len: int, batch: int = 1) -> torch.Tensor:
    """
    Return (3, batch, seq_len) int64 position IDs.
    The talker uses 3-D multimodal RoPE (temporal / height / width);
    for pure text / TTS all three dimensions share the same 1-D positions.
    """
    pos = torch.arange(seq_len, dtype=torch.long).view(1, 1, seq_len)
    return pos.expand(3, batch, seq_len).contiguous()


# ──────────────────────────────────────────────────────────────────────────────
# Component 1 : Speaker Encoder  (ECAPA-TDNN)
# ──────────────────────────────────────────────────────────────────────────────

class SpeakerEncoderWrapper(nn.Module):
    """
    Thin wrapper around Qwen3TTSSpeakerEncoder for tracing.

    Input:   mel_spectrogram  (1, T_frames, 128)   float32
    Output:  speaker_embedding (1, 1024)            float32

    Note: the underlying encoder immediately transposes to (B, 128, T) for its
    1-D conv stack, so we pass (B, T, 128) here to match the original API.
    """

    def __init__(self, speaker_encoder: nn.Module):
        super().__init__()
        self.enc = speaker_encoder

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.enc(mel)


def convert_speaker_encoder(
    full_model,
    output_dir: Path,
    target: ct.target,
    t_frames: int = 300,
) -> None:
    """
    Convert the ECAPA-TDNN speaker encoder to a CoreML package.

    t_frames: number of mel-spectrogram frames to trace at.
    At 16 kHz with hop_size=160, ~300 frames ≈ 3 seconds of audio.
    Choose a value that covers your longest reference clip.
    """
    print("\n[1/2] Converting Speaker Encoder …")

    if full_model.speaker_encoder is None:
        print("  ! Model has no speaker_encoder (not a 'base' type model). Skipping.")
        return

    wrapper = SpeakerEncoderWrapper(full_model.speaker_encoder).eval()
    example  = torch.zeros(1, t_frames, MEL_DIM)

    with torch.no_grad():
        # Verify it runs before tracing
        _ = wrapper(example)
        traced = torch.jit.trace(wrapper, example)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="mel_spectrogram",
                shape=(1, t_frames, MEL_DIM),
                dtype=np.float32,
            )
        ],
        outputs=[
            ct.TensorType(name="speaker_embedding", dtype=np.float32)
        ],
        minimum_deployment_target=target,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )

    out = output_dir / "SpeakerEncoder.mlpackage"
    mlmodel.save(str(out))
    print(f"  ✓ saved → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Component 2 : Talker  (full-context pass, no KV cache)
#
# WHAT THIS WRAPPER DOES:
#   • Calls the 20 Qwen3TTSTalkerDecoderLayer modules directly
#   • Supplies a pre-computed 4-D causal mask → skips create_causal_mask/vmap
#   • Passes past_key_values=None to each layer → caching disabled
#   • Has no branching, no Python objects, no .generate() calls
#
# TRADE-OFF:
#   Full O(S²) attention recompute every decode step.
#   Phase 2 will add a stateful KV-cache model for O(n) amortised cost.
# ──────────────────────────────────────────────────────────────────────────────

class TalkerContextWrapper(nn.Module):
    """
    Runs the full Talker decoder stack over a sequence and returns logits.

    Inputs:
      inputs_embeds  (1, S, HIDDEN_DIM)     float32
          Fused embeddings — built outside this module in Python/Swift:
          codec embeds + projected text embeds + speaker embed (if base model)
      causal_mask    (1, 1, S, S)           float32
          Pre-computed additive causal mask (0 = attend, -inf = mask).
          Use make_causal_mask(S, S) from this file.
      position_ids   (3, 1, S)              int32
          3-D multimodal RoPE positions.  Use make_3d_position_ids(S).

    Output:
      logits         (1, S, VOCAB_SIZE)     float32
          Unnormalised scores over the codec vocabulary for every position.
          For auto-regressive generation, sample from logits[:, -1, :].

    NOTE ON attn_implementation:
      Load the model with attn_implementation="eager" (see load_model() below).
      This ensures Qwen3TTSTalkerAttention uses eager_attention_forward —
      a plain matmul+softmax that CoreML handles well.  SDPA / Flash Attention
      ops are not supported by coremltools.
    """

    def __init__(self, talker_for_cond_gen: nn.Module):
        super().__init__()
        # talker_for_cond_gen  is  Qwen3TTSTalkerForConditionalGeneration
        # .model               is  Qwen3TTSTalkerModel
        model            = talker_for_cond_gen.model
        self.layers      = model.layers           # nn.ModuleList of decoder layers
        self.norm        = model.norm             # final RMSNorm
        self.rotary_emb  = model.rotary_emb       # Qwen3TTSTalkerRotaryEmbedding
        self.codec_head  = talker_for_cond_gen.codec_head  # Linear → vocab

    def forward(
        self,
        inputs_embeds: torch.Tensor,   # (1, S, hidden)
        causal_mask:   torch.Tensor,   # (1, 1, S, S)  pre-computed, no vmap needed
        position_ids:  torch.Tensor,   # (3, 1, S)     int64 / int32
    ) -> torch.Tensor:                 # (1, S, vocab)

        # position_ids must be int64 inside the model
        position_ids = position_ids.long()

        # Compute rotary (cos, sin) embeddings shared across all layers.
        # @dynamic_rope_update will use the cached inv_freq for these lengths.
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # text_position_ids: the [0] slice of the 3-D position_ids.
        # The attention modules receive this for any downstream use.
        text_position_ids = position_ids[0]                          # (1, S)

        # cache_position=None: only used when past_key_values is not None
        # (for cache position tracking in KV update).  Passing None avoids
        # torch.arange(tensor.shape[1]) which generates aten::size → int ops.

        hidden = inputs_embeds
        for layer in self.layers:
            # Calling the layer directly lets us supply:
            #   attention_mask = our pre-computed 4-D mask
            #   past_key_values = None  (no caching, pure full-attention)
            # This completely avoids Qwen3TTSTalkerModel.forward() and its
            # call to create_causal_mask / vmap.
            out    = layer(
                hidden,
                attention_mask      = causal_mask,     # 4-D → no vmap path
                position_ids        = text_position_ids,
                past_key_values     = None,            # disable KV cache
                output_attentions   = False,
                use_cache           = False,
                cache_position      = None,            # unused when past_kv=None; avoids aten::size→int
                position_embeddings = position_embeddings,
            )
            hidden = out[0]

        hidden = self.norm(hidden)
        return self.codec_head(hidden)                 # (1, S, vocab)


def convert_talker(
    full_model,
    output_dir: Path,
    target: ct.target,
    seq_len: int = DEFAULT_SEQ_LEN,
) -> None:
    """
    Convert the Talker to a CoreML package.

    seq_len: static sequence length to trace at.
    You can export multiple lengths (e.g. 64, 128, 256) for different prompt
    sizes and pick the right one at runtime.

    If you get a BlobWriter error here, the most common causes are:
      • An op in the traced graph that coremltools doesn't support yet.
        Run:  coremltools.utils.get_model_metadata(traced_mlmodel)
        to see which ops were converted, or use ct.convert(..., debug_mode=True).
      • torch 2.7.0 / coremltools 9.0 incompatibility.
        Fallback: torch==2.5.1 + coremltools==8.1.0 (see file header).
    """
    print(f"\n[2/2] Converting Talker (seq_len={seq_len}) …")

    # ── apply all CoreML patches BEFORE tracing ───────────────────────────────
    # Order matters: patches replace module-level symbols; trace captures them.
    patch_mrope_for_coreml()                     # fix split/in-place mRoPE ops
    patch_repeat_kv_and_rmsnorm_for_coreml()     # fix shape-unpack + dtype aten::Int ops

    wrapper = TalkerContextWrapper(full_model.talker).eval()

    # Example inputs
    ex_embeds  = torch.zeros(1, seq_len, HIDDEN_DIM)
    ex_mask    = make_causal_mask(seq_len, seq_len)              # (1,1,S,S) float32
    ex_pos_ids = make_3d_position_ids(seq_len)                   # (3,1,S) int64

    # Sanity check before tracing
    print(f"  Running forward pass smoke test …")
    with torch.no_grad():
        try:
            out = wrapper(ex_embeds, ex_mask, ex_pos_ids)
            print(f"  ✓ smoke test passed — output shape: {tuple(out.shape)}")
        except Exception as e:
            print(f"  ✗ smoke test failed: {e}")
            print("    Make sure the model was loaded with attn_implementation='eager'")
            raise

    # Trace
    print(f"  Tracing …")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (ex_embeds, ex_mask, ex_pos_ids))

    # Scan for remaining aten::Int ops before attempting conversion.
    # If count > 0, the listed nodes will cause the same "only 0-dimensional
    # arrays can be converted" error.  Each printed line shows the source
    # location and input type to guide the next patch.
    n_int_ops = print_int_ops(traced)
    if n_int_ops > 0:
        print(f"  ⚠  {n_int_ops} aten::Int op(s) remain — conversion will likely fail.")
        print("     See printed nodes above.  Add a patch in main.py to fix each one.")

    # Convert to CoreML
    print(f"  Converting to CoreML …")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="inputs_embeds",
                shape=(1, seq_len, HIDDEN_DIM),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="causal_mask",
                shape=(1, 1, seq_len, seq_len),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="position_ids",
                shape=(3, 1, seq_len),
                dtype=np.int32,
            ),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        minimum_deployment_target=target,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )

    out_path = output_dir / f"Talker_seq{seq_len}.mlpackage"
    mlmodel.save(str(out_path))
    print(f"  ✓ saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Input embedding helpers (used by the Swift/Python generation loop)
#
# These show how to build `inputs_embeds` that TalkerContextWrapper expects.
# These stay in Python for Phase 1; later you can CoreML-ise them too.
# ──────────────────────────────────────────────────────────────────────────────

def build_inputs_embeds(
    full_model,
    codec_token_ids:    torch.Tensor,   # (1, S) int64  — codec token IDs
    text_token_ids:     torch.Tensor,   # (1, S) int64  — text token IDs
    speaker_embedding:  torch.Tensor,   # (1, 1024) float32 (or None)
    is_tts_pad_mask:    torch.Tensor,   # (1, S) bool   — True where token is TTS-pad
) -> torch.Tensor:
    """
    Build fused input embeddings from codec + text token IDs.

    This replicates the embedding assembly that happens inside
    Qwen3TTSTalkerForConditionalGeneration.forward() during the prefill phase,
    extracted here so the CoreML model receives a single float tensor.

    For the actual token ID layout see Qwen3TTSForConditionalGeneration.generate().
    You will need to adapt this to match the exact prompt format your use-case needs.
    """
    talker = full_model.talker

    # Codec token embeddings  (1, S, hidden_dim)
    codec_embeds = talker.model.codec_embedding(codec_token_ids)

    # Text token embeddings projected to hidden_dim  (1, S, hidden_dim)
    text_embeds_raw = talker.model.text_embedding(text_token_ids)   # (1, S, text_hidden_dim)
    text_embeds     = talker.text_projection(text_embeds_raw)        # (1, S, hidden_dim)

    # Fused: add codec + text embeddings (they occupy non-overlapping positions
    # in the sequence; the mask tells us which is which)
    fused = codec_embeds + text_embeds

    # Optionally inject speaker embedding at the TTS-pad positions
    if speaker_embedding is not None:
        spk = speaker_embedding.unsqueeze(1)                         # (1, 1, hidden_dim)
        tts_pad_embed = full_model.talker.model.codec_embedding(
            torch.tensor([[full_model.config.talker_config.tts_pad_token_id]])
        )
        fused = torch.where(
            is_tts_pad_mask.unsqueeze(-1),
            tts_pad_embed + spk,
            fused,
        )

    return fused   # (1, S, hidden_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Generation loop sketch (Python / Swift pseudocode)
# ──────────────────────────────────────────────────────────────────────────────

def generation_loop_sketch(
    coreml_talker,          # loaded .mlpackage
    full_model,             # original PyTorch model (for sampling helpers)
    prompt_embeds: torch.Tensor,   # (1, P, 1024) — prefill embeddings
    max_new_tokens: int = 512,
) -> List[int]:
    """
    Illustrative Python generation loop.
    Replace with Swift code on-device.

    Phase 1 (no KV cache): append 1 token per step, re-run full context.
    Phase 2: use stateful KV-cache CoreML model (ct.StateType).
    """
    generated_ids: List[int] = []
    current_embeds = prompt_embeds          # grows by 1 each step

    for step in range(max_new_tokens):
        S = current_embeds.shape[1]
        mask    = make_causal_mask(S, S).numpy()
        pos_ids = make_3d_position_ids(S).numpy().astype(np.int32)

        # Run CoreML model
        result  = coreml_talker.predict({
            "inputs_embeds": current_embeds.numpy(),
            "causal_mask":   mask,
            "position_ids":  pos_ids,
        })
        logits  = torch.from_numpy(result["logits"])  # (1, S, vocab)
        next_logits = logits[0, -1, :]                # last position

        # Sample / greedy (keep in Python / Swift)
        next_id = int(torch.argmax(next_logits))      # greedy for illustration

        # Check EOS (replace with actual EOS token ID)
        eos_id = full_model.config.talker_config.eos_token_id
        if isinstance(eos_id, list):
            if next_id in eos_id:
                break
        elif next_id == eos_id:
            break

        generated_ids.append(next_id)

        # Append next token embedding for next step
        next_embed = full_model.talker.model.codec_embedding(
            torch.tensor([[next_id]])
        )                                             # (1, 1, hidden_dim)
        current_embeds = torch.cat([current_embeds, next_embed], dim=1)

    return generated_ids


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    """
    Load Qwen3TTS with settings suitable for CoreML export.

    attn_implementation="eager" is CRITICAL:
      • Forces Qwen3TTSTalkerAttention to use eager_attention_forward
        (plain matmul + softmax), which coremltools can convert.
      • Avoids SDPA / Flash Attention ops that are not in the CoreML op set.
      • Does NOT affect the vmap crash (that's in create_causal_mask, which
        our wrappers bypass entirely), but is still required for conversion.

    torch_dtype=torch.float32:
      Export in float32; coremltools will quantise to float16 at conversion
      time via compute_precision=ct.precision.FLOAT16.  Exporting in float16
      directly can cause precision issues during the trace.
    """
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

    print(f"Loading model from '{model_path}' …")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        attn_implementation="eager",   # disables SDPA — required for coremltools
    )
    model.eval()

    # Verify constants match the loaded config
    cfg = model.talker.config
    assert cfg.num_hidden_layers == NUM_TALKER_LAYERS, \
        f"NUM_TALKER_LAYERS mismatch: config={cfg.num_hidden_layers}, constant={NUM_TALKER_LAYERS}"
    assert cfg.num_key_value_heads == KV_HEADS, \
        f"KV_HEADS mismatch: config={cfg.num_key_value_heads}, constant={KV_HEADS}"
    assert cfg.head_dim == HEAD_DIM, \
        f"HEAD_DIM mismatch: config={cfg.head_dim}, constant={HEAD_DIM}"
    assert cfg.hidden_size == HIDDEN_DIM, \
        f"HIDDEN_DIM mismatch: config={cfg.hidden_size}, constant={HIDDEN_DIM}"

    print("  ✓ model loaded and config verified")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

_TARGET_MAP = {
    "ios16": ct.target.iOS16,
    "ios17": ct.target.iOS17,
    "ios18": ct.target.iOS18,
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-TTS to CoreML (.mlpackage)"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Local path or HuggingFace repo ID for the Qwen3-TTS weights",
    )
    parser.add_argument(
        "--output_dir",
        default="./coreml_models",
        help="Directory to write .mlpackage files into",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help=(
            "Static sequence length to trace the Talker at.  "
            "Must be >= your longest prompt.  Larger = more memory."
        ),
    )
    parser.add_argument(
        "--target",
        default="ios17",
        choices=list(_TARGET_MAP.keys()),
        help="Minimum iOS deployment target",
    )
    parser.add_argument(
        "--skip_speaker_encoder",
        action="store_true",
        help="Skip converting the speaker encoder (e.g. for custom_voice models)",
    )
    parser.add_argument(
        "--speaker_frames",
        type=int,
        default=300,
        help=(
            "Number of mel frames to trace the speaker encoder at.  "
            "At 16kHz / hop=160, 300 ≈ 3 seconds."
        ),
    )
    args = parser.parse_args()

    target     = _TARGET_MAP[args.target]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_model = load_model(args.model_path)

    if not args.skip_speaker_encoder:
        convert_speaker_encoder(
            full_model, output_dir, target, t_frames=args.speaker_frames
        )

    convert_talker(full_model, output_dir, target, seq_len=args.seq_len)

    print("\n──────────────────────────────────────────────")
    print("Conversion complete.  Next steps:")
    print()
    print("1. Validate each package:")
    print("     python -c \"")
    print("       import coremltools as ct")
    print("       m = ct.models.MLModel('coreml_models/Talker_seq256.mlpackage')")
    print("       print(m)\"")
    print()
    print("2. Test predictions (Python):")
    print("     result = m.predict({'inputs_embeds': …, 'causal_mask': …, 'position_ids': …})")
    print()
    print("3. Add stateful KV-cache (Phase 2) for efficient on-device inference.")
    print("   See https://apple.github.io/coremltools/docs-guides/source/stateful-models.html")
    print("──────────────────────────────────────────────")


if __name__ == "__main__":
    main()
