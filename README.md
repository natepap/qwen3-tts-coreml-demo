# Qwen3-TTS CoreML Demo

An on-device iOS demo that runs [Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) entirely through CoreML. This repo contains the PyTorch → CoreML conversion pipeline and the SwiftUI app that drives inference on-device.

---

## Repo structure

```
qwen3-tts-coreml-demo/
├── export_pipeline.py          # Main conversion script (run this to export models)
├── find_scatter_mapping.py     # Diagnostic: determines scatter output → KV layer mapping
├── pyproject.toml
├── coreml_conversion/          # Modular conversion library (alternative entry point)
│   ├── __init__.py
│   └── main.py
└── ios_app/                    # SwiftUI source files
    ├── TTSApp.swift
    ├── ContentView.swift
    └── TTSEngine.swift
```

---

## Setup

### 1. Clone the Qwen3-TTS model code

The export pipeline imports `qwen_tts` directly from the upstream repo.
Clone it as a sibling directory of this repo:

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git ../Qwen3-TTS
pip install -e ../Qwen3-TTS
```

### 2. Download model weights

Download the CustomVoice model checkpoint from Hugging Face.
The export pipeline expects it at `../Qwen3-TTS-12Hz-0.6B-CustomVoice/` by default (edit `MODEL_PATH` in `export_pipeline.py` to change).

```bash
# Using huggingface-cli
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --local-dir ../Qwen3-TTS-12Hz-0.6B-CustomVoice
```

### 3. Install Python dependencies

```bash
pip install -e .
```

> If you hit `BlobWriter` errors during CoreML conversion, see the fallback
> version stack in `pyproject.toml` comments. Do **not** downgrade
> `transformers` — the model code depends on APIs only present in 4.57+.

### 4. Run the export

```bash
python export_pipeline.py
```

This writes four `.mlpackage` files and Swift-ready embedding arrays into `qwen3_tts_test_pipeline/`:

| Output | Description |
|--------|-------------|
| `talker_prefill.mlpackage` | Full-context attention pass (no KV cache) |
| `talker_decode.mlpackage` | Single-token decode with explicit KV cache I/O |
| `code_predictor.mlpackage` | 15-codebook codec predictor |
| `speech_tokenizer_decode.mlpackage` | Vocoder (codec codes → waveform) |
| `swift_assets/` | `text_embeddings.npy`, `codec_embedding.npy`, `cp_codec_embedding_{0..14}.npy`, `config.json` |

### 5. iOS app

Create an Xcode project, add the Swift files from `ios_app/` to your app target, and add the swift-transformers package dependency for the Qwen2 BPE tokenizer. Symlink (or copy) the `.mlpackage` bundles and `swift_assets/` into the Xcode project folder so they are included as resources.

Minimum deployment target: **iOS 18** (required for the code predictor's `minimum_deployment_target=ct.target.iOS18`).

---

## How the pipeline works

### Why the model can't be exported as-is

`Qwen3TTSTalkerForConditionalGeneration.forward()` has three properties that make it un-exportable directly:

1. **Data-dependent branching** — `if inputs_embeds.shape[1] > 1` branches on a runtime tensor shape. `torch.export` / `jit.trace` cannot follow branches that depend on tensor values at trace time.

2. **Python object call inside `forward`** — `self.code_predictor.generate(...)` invokes a sampling loop that is a Python object, not a tensor operation. It cannot be traced.

3. **`vmap` inside `create_causal_mask`** — `transformers.masking_utils.create_causal_mask()` eventually calls `_vmap_for_bhqkv()`. During `torch.export` this raises:
   ```
   RuntimeError: Attempting to use FunctionalTensor on its own.
   Instead, please use it with FunctionalTensorMode()
   ```

**The fix:** thin `nn.Module` wrappers call the underlying transformer layers directly, accepting a pre-computed 4D float attention mask (`[1, 1, q_len, kv_len]`). This bypasses `create_causal_mask` entirely, removes all Python-level control flow, and gives coremltools a clean statically-shaped graph.

Two additional monkey-patches are applied before tracing:
- **mRoPE** — `apply_multimodal_rotary_pos_emb` uses `cos.split(mrope_section * 2)`, which generates `aten::split_with_sizes` with a 1-D tensor as sizes — coremltools can't lower this. Replaced with explicit Python-int slice indexing.
- **RMSNorm** — the original reads `hidden_states.dtype` at runtime, emitting a `prim::dtype` node that coremltools fails to lower. Replaced with an explicit `torch.float32` literal.

### Inference pipeline

```
User text
   │
   ▼  Qwen2 BPE tokenizer (swift-transformers)
Text token IDs
   │
   ▼  Lookup text_embeddings.npy  +  prepend tts_bos / codec_think tokens
inputs_embeds  [1, T, 1024]
   │
   ▼  talker_prefill.mlpackage  (padded to [1, 2048, 1024])
logits [1, 2048, 3072],  hidden_states [1, 2048, 1024]
   │
   ┌─── Prefill phase: run talker_decode token-by-token to populate KV cache
   │    (28 K arrays + 28 V arrays, each [1, 8, 2048, 128] fp16)
   │
   ▼  Autoregressive decode loop (until codec_eos token):
   │
   │   talker_decode.mlpackage  →  logits [1,1,3072],  hidden [1,1,1024]
   │   Sample cb0 from logits
   │   code_predictor.mlpackage →  cb1..cb15 (one per codebook head)
   │   Build next embed = sum(16 codec embeddings) + tts_pad_embed
   │
   ▼  Accumulate codec codes [1, 16, T_codec]
   │
   ▼  speech_tokenizer_decode.mlpackage  (chunked: 300 frames + 25 context)
   │   codes [1, 16, 325]  →  waveform [1, 1, 624000]
   │
   ▼  AVAudioEngine playback
```

**Key constants:**
- Template: `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
- Prefix tokens: `[151644, 77091, 198]`
- `tts_bos=151672`, `tts_eos=151673`, `tts_pad=151671`
- `codec_pad=2148`, `codec_bos=2149`, `codec_eos=2150`
- Vocoder: 12 Hz codec → 24 kHz audio (1920× upsample)

---

## Limitations

### The decoder KV cache problem (biggest issue)

`talker_decode.mlpackage` exports cleanly in PyTorch but has a critical issue at runtime in CoreML: when the model updates each layer's K and V cache via `scatter`, CoreML renames the outputs unpredictably. The outputs come out as `scatter`, `scatter_1`, `scatter_2`, ... with no guaranteed correspondence to which transformer layer they represent.

The `find_scatter_mapping.py` script was written to reverse-engineer the mapping by injecting sentinel values into one layer's cache at a time and observing which scatter output retains them. Even with this mapping, reliable round-tripping of 56 KV tensors (28 K + 28 V) per decode step proved too fragile in practice.

**Consequence:** the iOS app currently uses `talker_prefill.mlpackage` in a loop for the entire generation (both the initial context pass and all decode steps). This is **O(n²)** in attention cost — every new token re-processes the full context from scratch. For long outputs this is very slow.

The correct fix would be one of:
- Use `ct.StateType` (CoreML stateful KV cache, iOS 18+) to promote the KV tensors to in-model state, removing the input/output scatter round-trip entirely.
- Use the decode model with a confirmed scatter mapping and accept the 56-tensor I/O overhead.

### Other limitations

- **English only** — the model is Qwen3-TTS-12Hz-0.6B-**CustomVoice**; the language token is hardcoded to `english=2050`.
- **2048 token context limit** — the prefill model pads to a fixed 2048 × 1024 context window; inputs longer than ~2048 text tokens are truncated.
- **iOS 18+ required** — `code_predictor.mlpackage` targets iOS 18 (`ct.target.iOS18`); `talker_prefill` and the vocoder target iOS 17.
- **No voice cloning** — the speaker encoder (`SpeakerEncoder` / ECAPA-TDNN) is not wired into the iOS app; only the default voice is used.
- **Slow prefill** — without a stateful KV cache, long texts (>20 words) take 10–30 seconds on-device before audio begins. There is no streaming/sentence-level chunking implemented.
- **fp16 code predictor NaN** — the 5-layer code predictor produces NaN in fp16 (activations in the ~100 range overflow), so it runs in fp32. The other models run in fp16.
