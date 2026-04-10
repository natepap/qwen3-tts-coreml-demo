"""
Qwen3-TTS CoreML Export Pipeline
==================================

Converts Qwen3-TTS-12Hz-0.6B-CustomVoice to four CoreML .mlpackage files:

  1. talker_prefill.mlpackage  — full-context attention pass (no KV cache)
  2. talker_decode.mlpackage   — single-token decode with explicit KV cache I/O
  3. code_predictor.mlpackage  — 15-codebook predictor (left-padded, fixed len)
  4. speech_tokenizer_decode.mlpackage — vocoder (codes → waveform)

Also exports Swift-ready embedding matrices and a config.json.

Usage:
    python export_pipeline.py
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

from torch._decomp import core_aten_decompositions

import qwen_tts.core.models.modeling_qwen3_tts as _m
from qwen_tts import Qwen3TTSModel


# =============================================================================
# CONSTANTS
# =============================================================================

MODEL_PATH   = "../Qwen3-TTS-12Hz-0.6B-CustomVoice/"
OUT_DIR      = "qwen3_tts_test_pipeline"
SWIFT_ASSETS = os.path.join(OUT_DIR, "swift_assets")

SAMPLE_RATE = 24000

# Talker transformer dimensions (verified against configuration_qwen3_tts.py)
NUM_TALKER_LAYERS = 28
NUM_KV_HEADS      = 8
NUM_Q_HEADS       = 16
HEAD_DIM          = 128
HIDDEN            = 1024

# Sequence length constants
MAX_LEN    = 2048   # padded context window for prefill / decode KV cache
MAX_CP_LEN = 17     # code predictor: 1 (past_hidden) + 1 (cb0) + 15 (cb1..cb15)

# Speech tokenizer / vocoder chunking
CODEC_WINDOW  = 325   # total padded input window in codec frames
CODEC_CHUNK   = 300   # frames processed per vocoder chunk
CODEC_CONTEXT = 25    # left-context overlap between chunks

# mRoPE — read from model after loading; placeholders set below after load
MROPE_SECTION     = None   # e.g. [16, 24, 24]
MROPE_INTERLEAVED = False

N_REP = NUM_Q_HEADS // NUM_KV_HEADS   # 2


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def write_wav(filename: str, samples, sr: int = SAMPLE_RATE) -> None:
    """Write a float32 array as a 16-bit PCM WAV file into OUT_DIR."""
    samples = np.clip(np.array(samples, dtype=np.float32), -1.0, 1.0)
    n = len(samples)
    if not filename.startswith(OUT_DIR):
        filename = os.path.join(OUT_DIR, filename)
    with open(filename, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n * 2))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", n * 2))
        pcm = (samples * 32767).astype(np.int16)
        f.write(pcm.tobytes())


def make_causal_mask(q_len: int, kv_len: int,
                     dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """
    Return (1, 1, q_len, kv_len) additive causal mask.
    Attentable positions → 0.0, masked positions → -inf.
    Uses plain torch ops — no vmap, safe to call during tracing.
    """
    q_idx  = torch.arange(q_len,  dtype=torch.long).unsqueeze(1)
    kv_idx = torch.arange(kv_len, dtype=torch.long).unsqueeze(0)
    offset = kv_len - q_len
    attend = kv_idx <= q_idx + offset
    mask   = torch.where(attend,
                         torch.zeros(1, dtype=dtype),
                         torch.full((1,), float("-inf"), dtype=dtype))
    return mask.view(1, 1, q_len, kv_len)


def make_decode_mask_row(pos: int, max_len: int = MAX_LEN,
                         dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """
    Single causal mask row for decode step at position `pos`.
    Shape: (1, 1, 1, max_len).  0 for positions ≤ pos, -inf elsewhere.
    """
    valid = torch.arange(max_len) <= pos
    return torch.where(valid,
                       torch.zeros(max_len, dtype=dtype),
                       torch.full((max_len,), float("-inf"), dtype=dtype)
                       ).view(1, 1, 1, max_len)


def make_cp_mask(seq_len: int, max_len: int = MAX_CP_LEN) -> np.ndarray:
    """
    4D additive causal mask for a left-padded code-predictor sequence.
    Padding positions (before the seq_len real tokens) are fully masked.
    Returns float32 numpy array of shape (1, 1, max_len, max_len).
    """
    mask = np.full((1, 1, max_len, max_len), -np.inf, dtype=np.float32)
    pad = max_len - seq_len
    # Padding positions attend only to themselves — prevents all-inf softmax → NaN
    for i in range(pad):
        mask[0, 0, i, i] = 0.0
    # Real token rows: causal attention over real tokens only
    for i in range(seq_len):
        row = pad + i
        mask[0, 0, row, pad:pad + i + 1] = 0.0
    return mask


def build_position_ids(pos: int, device: str = "cpu") -> torch.Tensor:
    """3D mRoPE position IDs for a single decode token. Shape: (3, 1, 1)."""
    return torch.full((3, 1, 1), pos, dtype=torch.long, device=device)


def sample_token(logits: torch.Tensor, do_sample: bool,
                 temperature: float, top_k: int) -> int:
    """Sample (or greedy-decode) a single token index from logits."""
    logits = logits.float()
    if not do_sample:
        return logits.argmax().item()
    logits = logits / temperature
    if top_k > 0:
        top_vals, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
        logits[logits < top_vals[..., -1:]] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def remove_profiler_nodes(exported):
    """Remove torch profiler annotation nodes from an exported graph."""
    graph = exported.graph_module.graph
    for node in list(graph.nodes):
        if node.op == "call_function" and "profiler" in str(node.target).lower():
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
        elif node.op == "call_function" and "_record_function" in str(node.target).lower():
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
    graph.lint()
    exported.graph_module.recompile()
    return exported


def remove_identity_nodes(exported):
    """
    Remove alias, prims.view_of, and no-op aten.view nodes from the
    exported graph so coremltools receives a clean graph without identity ops.
    """
    graph = exported.graph_module.graph

    changed = True
    while changed:
        changed = False
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue

            target = str(node.target)

            # Pure identity ops — always safe to remove
            if any(x in target for x in ("alias", "prims.view_of", "prims::view_of")):
                node.replace_all_uses_with(node.args[0])
                graph.erase_node(node)
                changed = True
                continue

            # aten.view — only remove if it is a no-op (same shape)
            if "aten.view" in target and len(node.args) >= 2:
                input_node = node.args[0]
                if (
                    hasattr(input_node, "meta") and
                    hasattr(node, "meta") and
                    input_node.meta.get("val") is not None and
                    node.meta.get("val") is not None and
                    input_node.meta["val"].shape == node.meta["val"].shape
                ):
                    node.replace_all_uses_with(input_node)
                    graph.erase_node(node)
                    changed = True

    graph.lint()
    graph.eliminate_dead_code()
    exported.graph_module.recompile()
    return exported


def cast_all_graph_constants_to_fp16(exported):
    """
    Walk the exported graph and cast any float32 constant tensors to float16.
    This ensures weights embedded in the graph (e.g. from decompositions) do
    not force the CoreML compute graph back to float32.
    """
    graph = exported.graph_module.graph
    for node in list(graph.nodes):
        if node.op == "get_attr":
            # Retrieve the buffer / parameter tensor
            parts = node.target.split(".")
            obj = exported.graph_module
            for part in parts:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if isinstance(obj, torch.Tensor) and obj.dtype == torch.float32:
                obj.data = obj.data.to(torch.float16)
    exported.graph_module.recompile()
    return exported


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(dtype=torch.float16):
    """
    Load Qwen3-TTS with float16 weights and apply all CoreML-safe patches.

    Patches applied before any tracing:
      1. apply_multimodal_rotary_pos_emb → static-index, no-in-place version
      2. Qwen3TTSRMSNorm.forward         → explicit float32 literal (no prim::dtype)
    """
    global MROPE_SECTION, MROPE_INTERLEAVED

    print(f"Loading model from '{MODEL_PATH}' ...")
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        dtype=dtype,
    )
    m = tts.model
    m.eval()
    print("  Model loaded.")

    # Read mRoPE config from the first attention layer
    attn0 = m.talker.model.layers[0].self_attn
    MROPE_SECTION     = attn0.rope_scaling["mrope_section"]
    MROPE_INTERLEAVED = attn0.rope_scaling.get("interleaved", False)
    print(f"  mRoPE: section={MROPE_SECTION}, interleaved={MROPE_INTERLEAVED}")

    # --- Patch 1: CoreML-safe mRoPE -----------------------------------------
    # The original apply_multimodal_rotary_pos_emb uses:
    #   cos.split(mrope_section * 2, dim=-1)  → aten::split_with_sizes with a
    #   1-D int tensor, which coremltools can't lower to a scalar.
    # Fix: explicit Python-int slices per section.
    def _coreml_apply_multimodal_rotary_pos_emb(
        q, k, cos, sin, mrope_section, mrope_interleaved=False, unsqueeze_dim=1
    ):
        def _rotate_half(x):
            half = x.shape[-1] // 2
            return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

        if mrope_interleaved:
            half_dim  = sum(mrope_section)
            mod_num   = len(mrope_section)
            height_set = set(range(1, mrope_section[1] * mod_num, mod_num))
            width_set  = set(range(2, mrope_section[2] * mod_num, mod_num))
            is_height = torch.tensor(
                [i in height_set for i in range(half_dim)], dtype=torch.bool
            ).view(1, 1, half_dim)
            is_width = torch.tensor(
                [i in width_set for i in range(half_dim)], dtype=torch.bool
            ).view(1, 1, half_dim)

            def _interleave(x_in):
                x_h = x_in[..., :half_dim]
                mixed = torch.where(is_height, x_h[1],
                                    torch.where(is_width, x_h[2], x_h[0]))
                return torch.cat([mixed, mixed], dim=-1)

            cos_r = _interleave(cos).unsqueeze(unsqueeze_dim)
            sin_r = _interleave(sin).unsqueeze(unsqueeze_dim)
        else:
            sections = list(mrope_section) * 2
            starts   = []
            cursor   = 0
            for size in sections:
                starts.append(cursor)
                cursor += size
            cos_parts = []
            sin_parts = []
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

    _m.apply_multimodal_rotary_pos_emb = _coreml_apply_multimodal_rotary_pos_emb
    print("  Patched apply_multimodal_rotary_pos_emb (static indexing).")

    # --- Patch 2: CoreML-safe RMSNorm ----------------------------------------
    # The original uses hidden_states.dtype at runtime → prim::dtype node →
    # aten::Int on a 1-D array, which coremltools fails to lower.
    # Fix: explicit torch.float32 literal so no prim::dtype node is emitted.
    def _coreml_rmsnorm(self, hidden_states):
        orig_dtype = hidden_states.dtype
        h = hidden_states.float()
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.float() * h).to(orig_dtype)

    _m.Qwen3TTSRMSNorm.forward = _coreml_rmsnorm
    print("  Patched Qwen3TTSRMSNorm.forward (explicit float32 literal).")

    return tts, m


# =============================================================================
# WRAPPER CLASS DEFINITIONS
# =============================================================================

class TalkerWrapper(nn.Module):
    """
    Full-context prefill wrapper for the talker transformer.

    Accepts pre-built inputs_embeds (text + codec streams already summed) exactly
    as Qwen3TTSTalkerForConditionalGeneration.forward() constructs them, together
    with a pre-computed 4D additive causal mask (bypasses vmap / create_causal_mask).

    Returns (logits, hidden_states) so that hidden_states[:, -1] can seed the
    code predictor without a second forward pass.
    """
    def __init__(self, talker):
        super().__init__()
        self.backbone = talker.model
        self.head     = talker.codec_head

    def forward(self, inputs_embeds, attention_mask):
        # Explicit fp16 cast at the boundary ensures the export dtype is stable
        inputs_embeds = inputs_embeds.to(torch.float16)
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )
        hidden_states = outputs[0]
        logits        = self.head(hidden_states)
        return logits, hidden_states


class TalkerDecodeExportable(nn.Module):
    """
    Single-token decode wrapper with explicit KV-cache I/O for CoreML export.

    KV tensors are passed in as flat positional args (*kv_flat) so torch.export
    can trace them cleanly without in-place mutation.  After ct.convert,
    coremltools promotes the paired KV inputs/outputs to ct.StateType.

    Returns (logits, hidden, k_0_new, v_0_new, ..., k_27_new, v_27_new)
    with the updated KV tensors interleaved: k_0, v_0, k_1, v_1, ...
    """
    def __init__(self, talker):
        super().__init__()
        self.layers  = talker.model.layers
        self.norm    = talker.model.norm
        self.rotary  = talker.model.rotary_emb
        self.head    = talker.codec_head
        self.scaling = HEAD_DIM ** -0.5

    def forward(
        self,
        inputs_embeds,    # [1, 1, HIDDEN]
        causal_mask_row,  # [1, 1, 1, MAX_LEN]
        position_ids,     # [3, 1, 1]
        cache_pos,        # [1] int tensor
        *kv_flat,         # 56 tensors: k_0, v_0, k_1, v_1, ... each [1, 8, MAX_LEN, 128]
    ):
        cos, sin = self.rotary(inputs_embeds, position_ids)
        hidden   = inputs_embeds.clone()
        pos      = cache_pos[0]
        idx      = pos.view(1, 1, 1, 1).expand(1, NUM_KV_HEADS, 1, HEAD_DIM).long()

        new_ks, new_vs = [], []
        for i in range(NUM_TALKER_LAYERS):
            layer   = self.layers[i]
            attn    = layer.self_attn
            k_cache = kv_flat[i * 2]       # [1, 8, MAX_LEN, 128]
            v_cache = kv_flat[i * 2 + 1]   # [1, 8, MAX_LEN, 128]

            residual = hidden
            normed   = layer.input_layernorm(hidden)

            q = attn.q_norm(attn.q_proj(normed).view(1, 1, NUM_Q_HEADS, HEAD_DIM)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(normed).view(1, 1, NUM_KV_HEADS, HEAD_DIM)).transpose(1, 2)
            v = attn.v_proj(normed).view(1, 1, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

            q, k = _m.apply_multimodal_rotary_pos_emb(
                q, k, cos, sin, MROPE_SECTION, MROPE_INTERLEAVED
            )

            # Scatter-write new K/V into cache (non-in-place: returns new tensor)
            k_cache = k_cache.scatter(2, idx, k.to(torch.float16))
            v_cache = v_cache.scatter(2, idx, v.to(torch.float16))
            new_ks.append(k_cache)
            new_vs.append(v_cache)

            # GQA expand: [1, 8, MAX_LEN, 128] → [1, 16, MAX_LEN, 128]
            k_exp = k_cache.unsqueeze(2).expand(-1, -1, N_REP, -1, -1).flatten(1, 2)
            v_exp = v_cache.unsqueeze(2).expand(-1, -1, N_REP, -1, -1).flatten(1, 2)

            scores  = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scaling + causal_mask_row
            weights = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
            out     = torch.matmul(weights, v_exp).transpose(1, 2).reshape(1, 1, -1)
            hidden  = residual + attn.o_proj(out)
            hidden  = hidden + layer.mlp(layer.post_attention_layernorm(hidden))

        hidden = self.norm(hidden)
        # Interleave updated KV tensors: k_0, v_0, k_1, v_1, ...
        interleaved = [t for pair in zip(new_ks, new_vs) for t in pair]
        return (self.head(hidden), hidden, *interleaved)


class CodePredictorWrapper(nn.Module):
    """
    Fixed-length, left-padded wrapper for the 15-codebook code predictor.

    Input layout (left-padded to MAX_CP_LEN=17 positions):
      inputs_embeds:  [1, MAX_CP_LEN, 1024]
        — position 0 (or left-padded zeros): past_hidden from talker
        — position 1: talker.get_input_embeddings()(cb0_token_id)
        — positions 2..16: cp.model.codec_embedding[k](cb_{k+1})
      attention_mask: [1, 1, MAX_CP_LEN, MAX_CP_LEN]  (4D additive causal)

    Output: all_logits [15, 1, 2048]  — one set of logits per codebook head.
    Left-padding keeps the export shape fixed regardless of how many cb tokens
    have been sampled so far.
    """
    def __init__(self, code_predictor):
        super().__init__()
        self.model    = code_predictor.model
        self.lm_heads = code_predictor.lm_head

    def forward(self, inputs_embeds, attention_mask):
        outputs     = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )
        last_hidden = outputs[0][:, -1, :]   # [1, 1024]
        all_logits  = torch.stack(
            [h(last_hidden) for h in self.lm_heads], dim=0
        )                                     # [15, 1, 2048]
        return all_logits


class SpeechTokenizerDecodeWrapper(nn.Module):
    """
    Thin wrapper around Qwen3TTSTokenizerV2Decoder for CoreML export.

    The vocoder consumes codes of shape [1, num_quantizers, T] and returns
    a waveform tensor [1, 1, T * upsample_rate].  We fix T = CODEC_WINDOW (325
    frames) and run in float32 because the vocoder uses causal convolutions
    that are already numerically stable at fp32 precision.

    In the generation loop, chunked_decode() (defined in the model source) uses
    CODEC_CHUNK=300 frames per chunk with CODEC_CONTEXT=25 frames of left context
    to avoid boundary artefacts.  This wrapper exports the single-chunk forward.
    """
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, codes):
        # codes: [1, num_quantizers, CODEC_WINDOW]  int64
        return self.decoder(codes)


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def _get_full_decomp_table():
    """Return the full core-aten decomposition table used before ct.convert."""
    from torch._decomp import get_decompositions
    decomp_table = get_decompositions([
        torch.ops.aten.alias,
        torch.ops.prims.view_of,
    ])
    full = core_aten_decompositions()
    full.update(decomp_table)
    return full


def export_talker_prefill(m, captured):
    """
    Export TalkerWrapper (full-context pass, no KV cache) to CoreML.

    Inputs  : inputs_embeds [1, MAX_LEN, 1024] fp16
              attention_mask [1, 1, MAX_LEN, MAX_LEN] fp16
    Outputs : logits [1, MAX_LEN, 3072] fp16
              hidden_states [1, MAX_LEN, 1024] fp16
    """
    print("\n[1/4] Exporting talker_prefill.mlpackage ...")

    example_embeds = captured["inputs_embeds"].to(torch.float16)   # [1, T, 1024]
    T = example_embeds.shape[1]

    # Pad to fixed MAX_LEN
    example_embeds_padded = torch.zeros(1, MAX_LEN, HIDDEN, dtype=torch.float16)
    example_embeds_padded[:, :T] = example_embeds

    example_mask_padded = torch.full(
        (1, 1, MAX_LEN, MAX_LEN), float("-inf"), dtype=torch.float16
    )
    example_mask_padded[:, :, :T, :T] = make_causal_mask(T, T)

    wrapper = TalkerWrapper(m.talker)
    wrapper = wrapper.to(torch.float16)
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    full_decomp = _get_full_decomp_table()

    exported = torch.export.export(
        wrapper,
        (example_embeds_padded, example_mask_padded),
        strict=False,
    )
    exported = exported.run_decompositions(full_decomp)
    exported = remove_identity_nodes(exported)

    coreml = ct.convert(
        exported,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        inputs=[
            ct.TensorType(name="inputs_embeds",   shape=(1, MAX_LEN, HIDDEN),          dtype=np.float16),
            ct.TensorType(name="attention_mask",  shape=(1, 1, MAX_LEN, MAX_LEN),      dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
    )

    out_path = os.path.join(OUT_DIR, "talker_prefill.mlpackage")
    coreml.save(out_path)
    print(f"  Saved → {out_path}")
    return coreml


def export_talker_decode(m, captured):
    """
    Export TalkerDecodeExportable (single-token decode + explicit KV cache) to CoreML.

    Inputs  : inputs_embeds [1, 1, 1024] fp16
              causal_mask_row [1, 1, 1, MAX_LEN] fp16
              position_ids [3, 1, 1] int32
              cache_pos [1] int32
              k_0 .. k_27, v_0 .. v_27  each [1, 8, MAX_LEN, 128] fp16
    Outputs : logits [1, 1, 3072]
              hidden [1, 1, 1024]
              k_0_new .. k_27_new, v_0_new .. v_27_new  (interleaved)
    """
    print("\n[2/4] Exporting talker_decode.mlpackage ...")

    prefill_embeds = captured["inputs_embeds"]
    T_prefill = prefill_embeds.shape[1]

    example_embed     = prefill_embeds[:, -1:].to(torch.float16)
    example_mask_row  = make_decode_mask_row(T_prefill - 1, dtype=torch.float16)
    example_pos       = build_position_ids(T_prefill - 1)
    example_cache_pos = torch.tensor([T_prefill - 1])
    example_kv        = [
        torch.zeros(1, NUM_KV_HEADS, MAX_LEN, HEAD_DIM, dtype=torch.float16)
        for _ in range(NUM_TALKER_LAYERS * 2)
    ]

    decode_wrapper = TalkerDecodeExportable(m.talker)
    decode_wrapper = decode_wrapper.to(torch.float16)
    decode_wrapper.eval()
    for p in decode_wrapper.parameters():
        p.requires_grad_(False)

    exported_decode = torch.export.export(
        decode_wrapper,
        (example_embed, example_mask_row, example_pos, example_cache_pos, *example_kv),
        strict=False,
    )
    exported_decode = remove_profiler_nodes(exported_decode)
    exported_decode = exported_decode.run_decompositions({})
    exported_decode = remove_identity_nodes(exported_decode)

    kv_inputs = (
        [ct.TensorType(name=f"k_{i}", shape=(1, NUM_KV_HEADS, MAX_LEN, HEAD_DIM), dtype=np.float16)
         for i in range(NUM_TALKER_LAYERS)] +
        [ct.TensorType(name=f"v_{i}", shape=(1, NUM_KV_HEADS, MAX_LEN, HEAD_DIM), dtype=np.float16)
         for i in range(NUM_TALKER_LAYERS)]
    )

    coreml_decode = ct.convert(
        exported_decode,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        inputs=[
            ct.TensorType(name="inputs_embeds",   shape=(1, 1, HIDDEN),       dtype=np.float16),
            ct.TensorType(name="causal_mask_row", shape=(1, 1, 1, MAX_LEN),   dtype=np.float16),
            ct.TensorType(name="position_ids",    shape=(3, 1, 1),            dtype=np.int32),
            ct.TensorType(name="cache_pos",       shape=(1,),                 dtype=np.int32),
            *kv_inputs,
        ],
        compute_precision=ct.precision.FLOAT16,
    )

    out_path = os.path.join(OUT_DIR, "talker_decode.mlpackage")
    coreml_decode.save(out_path)
    print(f"  Saved → {out_path}")
    return coreml_decode


def export_code_predictor(m):
    """
    Export CodePredictorWrapper (left-padded, fixed-length) to CoreML.

    Inputs  : inputs_embeds  [1, MAX_CP_LEN, 1024]             fp32
              attention_mask [1, 1, MAX_CP_LEN, MAX_CP_LEN]    fp32
    Output  : all_logits     [15, 1, 2048]                     fp32

    Uses fp32 — the 5-layer transformer produces NaN in fp16 due to activation
    overflow with typical last_hidden values (~100 range).
    """
    print("\n[3/4] Exporting code_predictor.mlpackage ...")

    cp_wrapper = CodePredictorWrapper(m.talker.code_predictor)
    # keep fp32 — small model, no memory concern
    cp_wrapper.eval()
    for p in cp_wrapper.parameters():
        p.requires_grad_(False)

    example_cp_embeds = torch.zeros(1, MAX_CP_LEN, HIDDEN, dtype=torch.float32)
    example_cp_mask   = torch.zeros(1, 1, MAX_CP_LEN, MAX_CP_LEN, dtype=torch.float32)

    exported_cp = torch.export.export(
        cp_wrapper,
        (example_cp_embeds, example_cp_mask),
        strict=False,
    )
    exported_cp = remove_profiler_nodes(exported_cp)
    exported_cp = exported_cp.run_decompositions(torch.export.default_decompositions())
    exported_cp = remove_identity_nodes(exported_cp)

    coreml_cp = ct.convert(
        exported_cp,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        inputs=[
            ct.TensorType(name="inputs_embeds",  shape=(1, MAX_CP_LEN, HIDDEN),             dtype=np.float32),
            ct.TensorType(name="attention_mask", shape=(1, 1, MAX_CP_LEN, MAX_CP_LEN),      dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="all_logits", dtype=np.float32),
        ],
        compute_precision=ct.precision.FLOAT32,
    )

    out_path = os.path.join(OUT_DIR, "code_predictor.mlpackage")
    coreml_cp.save(out_path)
    print(f"  Saved → {out_path}")
    return coreml_cp


def export_speech_tokenizer(m):
    """
    Export SpeechTokenizerDecodeWrapper (vocoder) to CoreML.

    The decoder is exported in float32 (vocoder convolutions are fp32-stable).
    Input  : codes [1, num_quantizers, CODEC_WINDOW]  int32
    Output : waveform [1, 1, CODEC_WINDOW * upsample_rate]  float32

    At inference, call chunked_decode() in Python (or port the chunking loop to
    Swift) using CODEC_CHUNK=300 frames per chunk and CODEC_CONTEXT=25 frames of
    left context — this matches the .mlpackage window size of 325.
    """
    print("\n[4/4] Exporting speech_tokenizer_decode.mlpackage ...")

    decoder = m.speech_tokenizer.model.decoder
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)

    num_q = decoder.config.num_quantizers
    example_codes = torch.zeros(1, num_q, CODEC_WINDOW, dtype=torch.long)

    wrapper = SpeechTokenizerDecodeWrapper(decoder)
    wrapper.eval()

    with torch.no_grad():
        example_out = wrapper(example_codes)
    print(f"  Vocoder output shape: {tuple(example_out.shape)}")

    exported_st = torch.export.export(
        wrapper,
        (example_codes,),
        strict=False,
    )
    exported_st = remove_profiler_nodes(exported_st)
    exported_st = exported_st.run_decompositions(torch.export.default_decompositions())
    exported_st = remove_identity_nodes(exported_st)

    coreml_st = ct.convert(
        exported_st,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        inputs=[
            ct.TensorType(
                name="codes",
                shape=(1, num_q, CODEC_WINDOW),
                dtype=np.int32,
            )
        ],
        outputs=[
            ct.TensorType(name="waveform", dtype=np.float32),
        ],
        compute_precision=ct.precision.FLOAT32,
    )

    out_path = os.path.join(OUT_DIR, "speech_tokenizer_decode.mlpackage")
    coreml_st.save(out_path)
    print(f"  Saved → {out_path}")
    return coreml_st


# =============================================================================
# GENERATION FUNCTIONS
# =============================================================================

def capture_prefill_inputs(tts, m, text: str = "Hello World"):
    """
    Run the model's generate() just long enough to capture the inputs that
    Qwen3TTSTalkerForConditionalGeneration.generate() passes to talker.generate().
    Returns a dict with keys: inputs_embeds, attention_mask,
                               trailing_text_hidden, tts_pad_embed.
    """
    input_ids = tts._tokenize_texts([tts._build_assistant_text(text)])
    gen_kwargs = tts._merge_generate_kwargs()

    captured = {}
    orig_generate = m.talker.generate

    def _hook(inputs_embeds=None, attention_mask=None,
               trailing_text_hidden=None, tts_pad_embed=None, **kwargs):
        captured["inputs_embeds"]        = inputs_embeds.detach().clone()
        captured["attention_mask"]       = attention_mask.detach().clone()
        captured["trailing_text_hidden"] = trailing_text_hidden.detach().clone()
        captured["tts_pad_embed"]        = tts_pad_embed.detach().clone()
        raise StopIteration

    m.talker.generate = _hook
    try:
        with torch.no_grad():
            m.generate(input_ids=input_ids, languages=["english"],
                       non_streaming_mode=True, **gen_kwargs)
    except StopIteration:
        pass
    m.talker.generate = orig_generate

    print("Captured prefill inputs:")
    print("  inputs_embeds:        ", captured["inputs_embeds"].shape)
    print("  attention_mask:       ", captured["attention_mask"].shape)
    print("  trailing_text_hidden: ", captured["trailing_text_hidden"].shape)
    print("  tts_pad_embed:        ", captured["tts_pad_embed"].shape)
    return captured, gen_kwargs


def run_code_predictor_coreml(
    m, coreml_cp, last_hidden, cb0_embed,
    gen_kwargs, cb_so_far=None
):
    """
    Run the CoreML code predictor autoregressively for codebooks 1..15.

    Args:
        m           : full PyTorch model (for codec embeddings)
        coreml_cp   : loaded CoreML code_predictor.mlpackage
        last_hidden : [1, 1, HIDDEN] tensor (talker hidden state at last token)
        cb0_embed   : [1, 1, HIDDEN] tensor (talker embedding of cb0 token)
        gen_kwargs  : generation kwargs dict
        cb_so_far   : list of already-sampled cb tokens (len 0..14); starts empty

    Returns:
        List[int] of 15 codec tokens (cb1..cb15)
    """
    sub_sample = gen_kwargs.get("subtalker_dosample", True)
    sub_temp   = gen_kwargs.get("subtalker_temperature", 0.9)
    sub_top_k  = gen_kwargs.get("subtalker_top_k", 50)

    cp = m.talker.code_predictor

    # Build left-padded inputs_embeds and mask
    # Sequence so far: [past_hidden, cb0_embed, cb1_embed, ...]
    seq_tokens = [last_hidden, cb0_embed]
    if cb_so_far:
        for k, cb_tok in enumerate(cb_so_far):
            emb = cp.model.codec_embedding[k](
                torch.tensor([[cb_tok]])
            ).to(torch.float16)
            seq_tokens.append(emb)

    codec_tokens = []

    for k in range(15):
        seq_len = len(seq_tokens)

        # Left-pad to MAX_CP_LEN
        pad = MAX_CP_LEN - seq_len
        embeds_np = np.zeros((1, MAX_CP_LEN, HIDDEN), dtype=np.float32)
        for j, tok_emb in enumerate(seq_tokens):
            embeds_np[0, pad + j] = tok_emb.float().numpy().reshape(HIDDEN)

        mask_np = make_cp_mask(seq_len)  # already float32

        ct_out = coreml_cp.predict({
            "inputs_embeds":  embeds_np,
            "attention_mask": mask_np,
        })
        # all_logits output: [15, 1, 2048]
        all_logits_np = ct_out["all_logits"]   # named output from export
        
        cb_logits = torch.tensor(all_logits_np[k, 0])  # [2048]

        cb_next = sample_token(cb_logits, sub_sample, sub_temp, sub_top_k)
        codec_tokens.append(cb_next)

        if k < 14:
            emb = cp.model.codec_embedding[k](
                torch.tensor([[cb_next]])
            ).to(torch.float16)
            seq_tokens.append(emb)

    return codec_tokens


def generate_with_coreml(m, captured, coreml_talker, coreml_cp,
                          gen_kwargs, max_new_tokens: int = 500):
    """
    Autoregressive generation using the CoreML prefill talker and CoreML
    code predictor.

    Strategy:
      - Each step: pad cur_embeds to MAX_LEN, run coreml_talker to get
        hidden state and logits at the current last position.
      - Sample cb0 from talker logits, then run run_code_predictor_coreml
        for cb1..cb15.
      - Build next-step embedding as the sum of all 16 codec stream embeddings
        plus trailing_text_hidden or tts_pad_embed.

    Returns:
        torch.Tensor of shape [N, 16] — N generated codec frames.
    """
    talker = m.talker
    cp     = talker.code_predictor
    eos_id = m.config.talker_config.codec_eos_token_id

    do_sample   = gen_kwargs.get("do_sample", True)
    temperature = gen_kwargs.get("temperature", 0.9)
    top_k       = gen_kwargs.get("top_k", 50)

    cur_embeds = captured["inputs_embeds"]   # bfloat16, [1, T, 1024]
    trailing   = captured["trailing_text_hidden"]
    tts_pad    = captured["tts_pad_embed"]

    all_codec_ids = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            T = cur_embeds.shape[1]
            assert T <= MAX_LEN, f"Sequence length {T} exceeds MAX_LEN {MAX_LEN}"

            # Pad embeddings to MAX_LEN
            pad_embeds = np.zeros((1, MAX_LEN, HIDDEN), dtype=np.float32)
            pad_embeds[0, :T] = cur_embeds.float().numpy()

            # Build 4D additive causal mask with padding
            q_idx   = np.arange(MAX_LEN).reshape(-1, 1)
            kv_idx  = np.arange(MAX_LEN).reshape(1, -1)
            causal  = kv_idx <= q_idx                               # [MAX_LEN, MAX_LEN]
            valid   = np.zeros(MAX_LEN, dtype=bool)
            valid[:T] = True
            mask_4d = np.where(
                causal & valid[np.newaxis, :],
                0.0, -np.inf
            ).reshape(1, 1, MAX_LEN, MAX_LEN).astype(np.float32)

            ct_out = coreml_talker.predict({
                "inputs_embeds":  pad_embeds,
                "attention_mask": mask_4d,
            })

            # Identify logits (larger) and hidden (smaller) outputs by size
            outputs_by_size = sorted(ct_out.items(), key=lambda kv: kv[1].size)
            hidden_key = outputs_by_size[0][0]   # [1, MAX_LEN, 1024]
            logits_key = outputs_by_size[1][0]   # [1, MAX_LEN, 3072]

            last_hidden = torch.tensor(ct_out[hidden_key])[:, T-1:T].to(cur_embeds.dtype)
            last_logits = torch.tensor(ct_out[logits_key])[0, T-1].float()

            cb0 = sample_token(last_logits, do_sample, temperature, top_k)
            if cb0 == eos_id:
                print(f"  EOS at step {step}")
                break

            # Embed cb0 using talker's embedding table
            cb0_embed = talker.get_input_embeddings()(
                torch.tensor([[cb0]], device=cur_embeds.device)
            ).to(torch.float16)

            # Run CoreML code predictor for cb1..cb15
            cb_rest = run_code_predictor_coreml(
                m, coreml_cp, last_hidden.to(torch.float16), cb0_embed,
                gen_kwargs
            )
            codec_tokens = [cb0] + cb_rest
            all_codec_ids.append(codec_tokens)

            # Build next-step embedding: sum of all 16 codec stream embeddings
            codec_sum = cb0_embed.clone()
            for i in range(15):
                codec_sum = codec_sum + cp.model.codec_embedding[i](
                    torch.tensor([[codec_tokens[i + 1]]], device=cur_embeds.device)
                )
            # Add trailing text hidden or TTS pad embedding
            if step < trailing.shape[1]:
                codec_sum = codec_sum + trailing[:, step:step + 1]
            else:
                codec_sum = codec_sum + tts_pad

            cur_embeds = torch.cat([cur_embeds, codec_sum], dim=1)

    return torch.tensor(all_codec_ids)   # [N, 16]


def decode_speech_coreml(coreml_st, codes, decode_upsample_rate, num_quantizers=16):
    """
    Decode codec frames to audio using the CoreML speech tokenizer.

    Mirrors chunked_decode() from the original model:
      - Process in CODEC_CHUNK-frame windows with CODEC_CONTEXT frames of left context.
      - Discard the first CODEC_CONTEXT * decode_upsample_rate samples from each
        chunk output (left-context artefacts).
      - Concatenate chunks to produce the full waveform.

    Args:
        coreml_st          : loaded CoreML speech_tokenizer_decode.mlpackage
        codes              : np.ndarray [N, 16] int32 codec tokens
        decode_upsample_rate: int, samples per codec frame (e.g. 2000 for 12Hz→24kHz)
        num_quantizers     : int, number of codec streams (default 16)

    Returns:
        np.ndarray: float32 waveform, shape [total_samples]
    """
    N = codes.shape[0]                          # total codec frames
    # codes: [N, 16] → [1, 16, N]
    codes_t = codes.T[np.newaxis]               # [1, 16, N]

    wav_chunks = []
    start = 0
    while start < N:
        end          = min(start + CODEC_CHUNK, N)
        context      = min(CODEC_CONTEXT, start)   # no left context on first chunk
        chunk_codes  = codes_t[:, :, start - context : end]  # [1, 16, chunk_len]
        chunk_len    = chunk_codes.shape[-1]

        # Left-pad with zeros to CODEC_WINDOW if shorter
        if chunk_len < CODEC_WINDOW:
            pad_codes = np.zeros((1, num_quantizers, CODEC_WINDOW), dtype=np.int32)
            pad_codes[:, :, :chunk_len] = chunk_codes
        else:
            pad_codes = chunk_codes.astype(np.int32)

        ct_out   = coreml_st.predict({"codes": pad_codes})
        wav_full = ct_out["waveform"].squeeze()              # [CODEC_WINDOW * upsample]

        # Discard left-context portion and any padding tail
        discard  = context * decode_upsample_rate
        keep     = (end - start) * decode_upsample_rate
        wav_chunks.append(wav_full[discard : discard + keep])
        start = end

    return np.concatenate(wav_chunks).astype(np.float32)


def generate_end_to_end_coreml(tts, m, text, coreml_prefill, coreml_cp, coreml_st,
                                 max_new_tokens=500):
    """
    Fully CoreML end-to-end pipeline:
      1. Tokenize text and capture prefill inputs via hook.
      2. Autoregressive talker + code predictor (both CoreML).
      3. Speech tokenizer decode (CoreML).

    Returns:
        np.ndarray: float32 audio waveform at SAMPLE_RATE Hz.
    """
    # Step 1: capture prefill inputs for this text
    captured, gen_kwargs = capture_prefill_inputs(tts, m, text=text)

    # Step 2: generate codec frames
    print(f"Generating codec frames (max {max_new_tokens} steps) ...")
    t0 = time.time()
    codes = generate_with_coreml(
        m, captured, coreml_prefill, coreml_cp, gen_kwargs,
        max_new_tokens=max_new_tokens,
    )
    elapsed = time.time() - t0
    print(f"  {codes.shape[0]} frames in {elapsed:.1f}s  "
          f"({codes.shape[0] / elapsed:.1f} frames/s)")

    if codes.shape[0] == 0:
        print("  No frames generated.")
        return np.zeros(0, dtype=np.float32)

    # Step 3: decode to audio with CoreML speech tokenizer
    decode_upsample_rate = int(m.speech_tokenizer.model.decode_upsample_rate)
    print(f"Decoding {codes.shape[0]} frames → audio "
          f"(upsample ×{decode_upsample_rate}) ...")
    t1 = time.time()
    audio = decode_speech_coreml(
        coreml_st, codes.numpy().astype(np.int32),
        decode_upsample_rate=decode_upsample_rate,
    )
    print(f"  {len(audio)/SAMPLE_RATE:.2f}s audio in {time.time()-t1:.1f}s")
    return audio


# =============================================================================
# SWIFT ASSET EXPORT
# =============================================================================

def export_swift_assets(m):
    """
    Save embedding matrices and a config.json that the Swift app reads at
    startup to avoid re-importing the full PyTorch model on-device.

    Files written to SWIFT_ASSETS:
      text_embeddings.npy          — talker text embedding + projection weights
      codec_embedding.npy          — talker codec embedding (cb0, 3072-vocab)
      cp_codec_embedding_{0..14}.npy — code predictor codec embeddings
      config.json                  — scalar dimensions and token IDs
    """
    ensure_dir(SWIFT_ASSETS)
    print(f"\nExporting Swift assets to '{SWIFT_ASSETS}' ...")

    talker = m.talker
    cp     = talker.code_predictor

    # Text embeddings: run embedding table through text_projection MLP and save
    # the result so Swift only needs a single row-lookup per token.
    with torch.no_grad():
        text_emb_weight  = talker.model.text_embedding.weight  # [vocab_text, hidden_in]
        text_proj_embeds = talker.text_projection(text_emb_weight).float()     # [vocab_text, hidden_out]
    np.save(os.path.join(SWIFT_ASSETS, "text_embeddings.npy"),
            text_proj_embeds.numpy())
    print(f"  text_embeddings.npy  {tuple(text_proj_embeds.shape)}")

    # Talker codec embedding (single Embedding(3072, 1024))
    with torch.no_grad():
        codec_emb = talker.model.codec_embedding.weight.float()        # [3072, 1024]
    np.save(os.path.join(SWIFT_ASSETS, "codec_embedding.npy"),
            codec_emb.numpy())
    print(f"  codec_embedding.npy  {tuple(codec_emb.shape)}")

    # Code predictor codec embeddings (cb1..cb15, 2048-vocab each)
    for k in range(15):
        with torch.no_grad():
            emb = cp.model.codec_embedding[k].weight.float()           # [2048, 1024]
        fname = f"cp_codec_embedding_{k}.npy"
        np.save(os.path.join(SWIFT_ASSETS, fname), emb.numpy())
    print(f"  cp_codec_embedding_{{0..14}}.npy  {tuple(emb.shape)} each")

    # Config JSON
    upsample_rate = int(np.prod(
        m.speech_tokenizer.model.decoder.config.upsample_rates +
        m.speech_tokenizer.model.decoder.config.upsampling_ratios
    ))
    config = {
        "hidden_dim":          HIDDEN,
        "num_talker_layers":   NUM_TALKER_LAYERS,
        "num_kv_heads":        NUM_KV_HEADS,
        "num_q_heads":         NUM_Q_HEADS,
        "head_dim":            HEAD_DIM,
        "max_len":             MAX_LEN,
        "max_cp_len":          MAX_CP_LEN,
        "codec_window":        CODEC_WINDOW,
        "codec_chunk":         CODEC_CHUNK,
        "codec_context":       CODEC_CONTEXT,
        "decode_upsample_rate": upsample_rate,
        "output_sample_rate":  SAMPLE_RATE,
        "codec_vocab_size":    int(cp.model.codec_embedding[0].weight.shape[0]),
        "num_codec_streams":   16,
        "talker_vocab_size":   int(talker.model.codec_embedding.weight.shape[0]),
        "codec_embed_size":    int(talker.model.codec_embedding.weight.shape[0]),
        "eos_token_id":        int(m.config.talker_config.codec_eos_token_id),
    }
    config_path = os.path.join(SWIFT_ASSETS, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  config.json")
    print(f"  Done. EOS token id = {config['eos_token_id']}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    ensure_dir(OUT_DIR)

    # ── Load model and apply patches ──────────────────────────────────────────
    tts, m = load_model()

    # ── Capture prefill inputs via hook ──────────────────────────────────────
    captured, gen_kwargs = capture_prefill_inputs(tts, m, text="Hello World")

    # # ── Export models ─────────────────────────────────────────────────────────
    coreml_prefill  = export_talker_prefill(m, captured)
    coreml_decode   = export_talker_decode(m, captured)
    coreml_cp       = export_code_predictor(m)
    tts32, m32 = load_model(torch.float32) # speech tokenizer requires fp32 loaded model
    coreml_st       = export_speech_tokenizer(m32)

    # # ── Export Swift assets ───────────────────────────────────────────────────
    # export_swift_assets(m)

    # ── Reload models with CPU+GPU only (ANE unsupported ops) ────────────────
    print("\nLoading models for inference (CPU+GPU) ...")
    coreml_prefill = ct.models.MLModel(
        os.path.join(OUT_DIR, "talker_prefill.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    coreml_cp = ct.models.MLModel(
        os.path.join(OUT_DIR, "code_predictor.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    coreml_st = ct.models.MLModel(
        os.path.join(OUT_DIR, "speech_tokenizer_decode.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )

    # ── Full end-to-end CoreML generation ────────────────────────────────────
    print("\nRunning end-to-end CoreML generation ...")
    audio = generate_end_to_end_coreml(
        tts, m,
        text="Hello World",
        coreml_prefill=coreml_prefill,
        coreml_cp=coreml_cp,
        coreml_st=coreml_st,
        max_new_tokens=500,
    )
    if len(audio) > 0:
        write_wav("coreml_end_to_end.wav", audio)
        print(f"  Saved → {OUT_DIR}/coreml_end_to_end.wav  ({len(audio)/SAMPLE_RATE:.2f}s)")

    print("\nExport pipeline complete.")
    print(f"  talker_prefill.mlpackage         → {OUT_DIR}/talker_prefill.mlpackage")
    print(f"  talker_decode.mlpackage          → {OUT_DIR}/talker_decode.mlpackage")
    print(f"  code_predictor.mlpackage         → {OUT_DIR}/code_predictor.mlpackage")
    print(f"  speech_tokenizer_decode.mlpackage→ {OUT_DIR}/speech_tokenizer_decode.mlpackage")
    print(f"  Swift assets                     → {SWIFT_ASSETS}/")
