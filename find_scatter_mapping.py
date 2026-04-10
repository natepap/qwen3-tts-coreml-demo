"""
find_scatter_mapping.py
=======================
Determines the true mapping between CoreML scatter output names and KV cache layers.

Strategy: pass a UNIQUE large value in k_cache[layer_i] for one layer at a time.
The scatter output that contains those values = k output for layer_i.
Similarly for v_cache.

Also verifies whether the prefill model on CPU_AND_GPU is producing correct output.

Run from: Qwen3-TTS/
    python find_scatter_mapping.py
"""

import os
import numpy as np
import coremltools as ct

OUT_DIR = "qwen3_tts_test_pipeline"
ASSETS  = os.path.join(OUT_DIR, "swift_assets")

NUM_TALKER_LAYERS = 28
NUM_KV_HEADS      = 8
HEAD_DIM          = 128
MAX_LEN           = 2048
HIDDEN            = 1024

codec_emb = np.load(os.path.join(ASSETS, "codec_embedding.npy"))
C_BOS = 2149
embed_in = codec_emb[C_BOS].astype(np.float16).reshape(1, 1, HIDDEN)

print("Loading decode (CPU_ONLY) ...")
decode = ct.models.MLModel(os.path.join(OUT_DIR, "talker_decode.mlpackage"),
                           compute_units=ct.ComputeUnit.CPU_ONLY)

def make_mask_row(pos):
    row = np.full((1, 1, 1, MAX_LEN), -np.inf, dtype=np.float16)
    row[0, 0, 0, :pos+1] = 0.0
    return row

def flat_kv_inputs(kc, vc):
    """Current interleaved mapping: k_m and v_j map into kv_flat positionally."""
    d = {}
    for m in range(NUM_TALKER_LAYERS):
        layer = m // 2
        d[f"k_{m}"] = kc[layer] if m % 2 == 0 else vc[layer]
    for j in range(NUM_TALKER_LAYERS):
        flat = NUM_TALKER_LAYERS + j
        layer = flat // 2
        d[f"v_{j}"] = kc[layer] if flat % 2 == 0 else vc[layer]
    return d

def simple_kv_inputs(kc, vc):
    """Simple mapping: k_i → kc[i], v_i → vc[i]."""
    d = {}
    for i in range(NUM_TALKER_LAYERS):
        d[f"k_{i}"] = kc[i]
        d[f"v_{i}"] = vc[i]
    return d

# ── STEP 1: Verify prefill on CPU_ONLY vs CPU_AND_GPU ────────────────────────
print("\n[STEP 1] Comparing prefill CPU_ONLY vs CPU_AND_GPU ...")
import importlib.util
spec = importlib.util.spec_from_file_location("ep", "export_pipeline.py")
ep   = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ep)
from qwen_tts import Qwen3TTSModel
tts = Qwen3TTSModel.from_pretrained("../Qwen3-TTS-12Hz-0.6B-CustomVoice/")
captured, _ = ep.capture_prefill_inputs(tts, tts, text="Hello World")
embeds = captured["inputs_embeds"][0].float().numpy()
T = embeds.shape[0]

pad_ctx = np.zeros((1, MAX_LEN, HIDDEN), dtype=np.float16)
pad_ctx[0, :T] = embeds.astype(np.float16)
q_idx  = np.arange(MAX_LEN).reshape(-1, 1)
kv_idx = np.arange(MAX_LEN).reshape(1, -1)
valid  = np.zeros(MAX_LEN, dtype=bool); valid[:T] = True
mask_4d = np.where((kv_idx <= q_idx) & valid[np.newaxis, :], 0.0, -np.inf
                   ).reshape(1, 1, MAX_LEN, MAX_LEN).astype(np.float16)

for cu_name, cu in [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
                     ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU)]:
    pf = ct.models.MLModel(os.path.join(OUT_DIR, "talker_prefill.mlpackage"),
                           compute_units=cu)
    out = pf.predict({"inputs_embeds": pad_ctx, "attention_mask": mask_4d})
    keys = sorted(out.keys(), key=lambda k: out[k].size)
    logits = out[keys[1]][0, T-1].astype(np.float32)
    top3 = np.argsort(logits)[-3:][::-1]
    print(f"  Prefill {cu_name}: top-3 = {[(int(i), round(float(logits[i]),2)) for i in top3]}  max={logits.max():.2f}")

# ── STEP 2: Find scatter output → layer mapping ───────────────────────────────
print("\n[STEP 2] Finding scatter output → layer mapping ...")
print("  Passing SENTINEL value (999.0) in k_cache[layer] one layer at a time")
print("  Checking which scatter output retains sentinel values ...\n")

SENTINEL = 999.0
shape = (1, NUM_KV_HEADS, MAX_LEN, HEAD_DIM)

# Run with all-zero caches first (baseline)
kc_zero = [np.zeros(shape, dtype=np.float16) for _ in range(NUM_TALKER_LAYERS)]
vc_zero = [np.zeros(shape, dtype=np.float16) for _ in range(NUM_TALKER_LAYERS)]
inp_base = {
    "inputs_embeds":   embed_in,
    "causal_mask_row": make_mask_row(0),
    "position_ids":    np.full((3, 1, 1), 0, dtype=np.int32),
    "cache_pos":       np.array([0], dtype=np.int32),
    **simple_kv_inputs(kc_zero, vc_zero),
}
out_base = decode.predict(inp_base)

# For each layer 0, 1, 2 test which scatter output changes
k_output_map = {}   # layer → scatter output name
v_output_map = {}

for test_layer in range(3):   # only need first 3 to find the pattern
    # Mark k_cache[test_layer] with sentinel
    kc = [np.zeros(shape, dtype=np.float16) for _ in range(NUM_TALKER_LAYERS)]
    kc[test_layer][:] = np.float16(SENTINEL)
    inp = {
        "inputs_embeds":   embed_in,
        "causal_mask_row": make_mask_row(0),
        "position_ids":    np.full((3, 1, 1), 0, dtype=np.int32),
        "cache_pos":       np.array([0], dtype=np.int32),
        **simple_kv_inputs(kc, vc_zero),
    }
    out = decode.predict(inp)
    # Find which scatter outputs differ from baseline (ignoring position 0 which was overwritten)
    changed = []
    for sname in [k for k in out.keys() if k.startswith("scatter")]:
        diff = np.abs(out[sname].astype(np.float32) - out_base[sname].astype(np.float32))
        # Check positions > 0 (pos 0 is always overwritten by scatter)
        if diff[:, :, 1:, :].max() > 1.0:
            changed.append((sname, diff.max()))
    changed.sort(key=lambda x: -x[1])
    print(f"  k_cache[{test_layer}] → changed scatter outputs: {changed[:3]}")
    if changed:
        k_output_map[test_layer] = changed[0][0]

    # Mark v_cache[test_layer] with sentinel
    vc = [np.zeros(shape, dtype=np.float16) for _ in range(NUM_TALKER_LAYERS)]
    vc[test_layer][:] = np.float16(SENTINEL)
    inp = {
        "inputs_embeds":   embed_in,
        "causal_mask_row": make_mask_row(0),
        "position_ids":    np.full((3, 1, 1), 0, dtype=np.int32),
        "cache_pos":       np.array([0], dtype=np.int32),
        **simple_kv_inputs(kc_zero, vc),
    }
    out = decode.predict(inp)
    changed = []
    for sname in [k for k in out.keys() if k.startswith("scatter")]:
        diff = np.abs(out[sname].astype(np.float32) - out_base[sname].astype(np.float32))
        if diff[:, :, 1:, :].max() > 1.0:
            changed.append((sname, diff.max()))
    changed.sort(key=lambda x: -x[1])
    print(f"  v_cache[{test_layer}] → changed scatter outputs: {changed[:3]}")
    if changed:
        v_output_map[test_layer] = changed[0][0]

print(f"\n  k output map (first 3): {k_output_map}")
print(f"  v output map (first 3): {v_output_map}")

# ── STEP 3: Use correct mapping for full decode comparison ────────────────────
print("\n[STEP 3] Re-running decode comparison with correct output mapping ...")

# Infer full pattern from the first 3 layers
def infer_name(layer, is_k, k_map, v_map):
    """Extrapolate pattern from known entries."""
    # Try to find pattern: look at the numbers in the known names
    ref_map = k_map if is_k else v_map
    if layer in ref_map:
        return ref_map[layer]
    # Extrapolate: find numeric pattern from layer 0, 1, 2
    nums = []
    for l in sorted(ref_map.keys()):
        name = ref_map[l]
        n = 0 if name == "scatter" else int(name.split("_")[1])
        nums.append((l, n))
    if len(nums) >= 2:
        step = nums[1][1] - nums[0][1]
        base_num = nums[0][1] - nums[0][0] * step
        n = base_num + layer * step
        return "scatter" if n == 0 else f"scatter_{n}"
    return None

# Build full read_kv_out using inferred pattern
def read_kv_correct(out, kc, vc):
    for layer in range(NUM_TALKER_LAYERS):
        kname = infer_name(layer, True, k_output_map, v_output_map)
        vname = infer_name(layer, False, k_output_map, v_output_map)
        if kname and kname in out: kc[layer] = out[kname]
        if vname and vname in out: vc[layer] = out[vname]
    return kc, vc

# Now redo the decode comparison
pf = ct.models.MLModel(os.path.join(OUT_DIR, "talker_prefill.mlpackage"),
                       compute_units=ct.ComputeUnit.CPU_ONLY)
pf_out = pf.predict({"inputs_embeds": pad_ctx, "attention_mask": mask_4d})
pf_keys = sorted(pf_out.keys(), key=lambda k: pf_out[k].size)
pf_logits = pf_out[pf_keys[1]][0, T-1].astype(np.float32)
top5_pf = np.argsort(pf_logits)[-5:][::-1]
print(f"  Prefill (CPU_ONLY) top-5: {[(int(i), round(float(pf_logits[i]),2)) for i in top5_pf]}")

kc, vc = [np.zeros(shape, dtype=np.float16) for _ in range(NUM_TALKER_LAYERS)], \
         [np.zeros(shape, dtype=np.float16) for _ in range(NUM_TALKER_LAYERS)]
for pos in range(T):
    inp = {
        "inputs_embeds":   embeds[pos:pos+1].astype(np.float16).reshape(1, 1, HIDDEN),
        "causal_mask_row": make_mask_row(pos),
        "position_ids":    np.full((3, 1, 1), pos, dtype=np.int32),
        "cache_pos":       np.array([pos], dtype=np.int32),
        **simple_kv_inputs(kc, vc),
    }
    out = decode.predict(inp)
    kc, vc = read_kv_correct(out, kc, vc)
    if pos % 4 == 0: print(f"  decode pos {pos}/{T}", end="\r", flush=True)
print()

dc_logits = out["linear_196"][0, 0].astype(np.float32)
top5_dc = np.argsort(dc_logits)[-5:][::-1]
print(f"  Decode top-5: {[(int(i), round(float(dc_logits[i]),2)) for i in top5_dc]}")
overlap = len(set(top5_pf.tolist()) & set(top5_dc.tolist()))
print(f"  Overlap: {overlap}/5")
