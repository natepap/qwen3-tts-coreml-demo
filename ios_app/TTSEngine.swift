// TTSEngine.swift
// Qwen3-TTS CoreML inference engine for iOS
//
// Pipeline (non-streaming, English):
//   1. Tokenise text with Qwen2 BPE tokenizer
//   2. Build inputs_embeds from text_embeddings.npy + codec_embedding.npy
//   3. Prefill phase: run talker_decode token-by-token over initial context,
//      populating KV cache (k_0..k_27, v_0..v_27)
//   4. Autoregressive decode:
//        a. talker_decode.mlpackage (single token, O(1)) → logits, hidden
//        b. Sample cb0 from talker logits
//        c. code_predictor.mlpackage → cb1..cb15
//        d. Build next embed = sum(16 codec embeds) + tts_pad_embed
//   5. speech_tokenizer_decode.mlpackage  → waveform (chunked)
//   6. Build WAV and play via AVFoundation

import Foundation
import CoreML
import AVFoundation
import Combine
import Tokenizers   // huggingface/swift-transformers

// MARK: - Model dimensions & token IDs (from config.json + model config)

private let kMaxLen        = 2048
private let kHidden        = 1024
private let kNumLayers     = 28      // talker transformer layers
private let kNumKVHeads    = 8       // KV heads per layer
private let kHeadDim       = 128     // head dimension
private let kMaxCpLen      = 17
private let kCodecWindow   = 325
private let kCodecChunk    = 300
private let kCodecContext  = 25
private let kUpsample      = 1920    // frames → samples (12 Hz → 24 kHz)
private let kSampleRate    = 24000
private let kNumStreams    = 16
private let kTalkerVocab   = 3072
private let kCodecVocab    = 2048
private let kEosId         = 2150

// Text-space special tokens (tts_bos/eos/pad are in the Qwen text vocab)
private let kTtsBosId  = 151672
private let kTtsEosId  = 151673
private let kTtsPadId  = 151671

// Talker codec-space special tokens
private let kCPadId    = 2148   // codec_pad_id
private let kCBosId    = 2149   // codec_bos_id
private let kCThinkId  = 2154   // codec_think_id
private let kCTBosId   = 2156   // codec_think_bos_id
private let kCTEosId   = 2157   // codec_think_eos_id
private let kCEnglish  = 2050   // language_id["english"]

// Role tokens: <|im_start|>=151644, assistant=77091, \n=198
private let kRoleTokens = [151644, 77091, 198]

// Sampling defaults
private let kDoSample    = true
private let kTemp        = Float(0.9)
private let kTopK        = 50
private let kSubSample   = true
private let kSubTemp     = Float(0.9)
private let kSubTopK     = 50

// MARK: - NPY row-reader (memory-mapped, float32)

struct NpyRowReader {
    private let data: Data
    let rows: Int
    let cols: Int
    private let dataOffset: Int

    init(url: URL) throws {
        data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count > 10,
              data[0] == 0x93,
              data[1] == 0x4e, data[2] == 0x55, data[3] == 0x4d,
              data[4] == 0x50, data[5] == 0x59
        else { throw TTSError.npyFormat("bad magic in \(url.lastPathComponent)") }

        let major = data[6]
        let headerLen: Int
        let headerStart: Int
        if major == 1 {
            headerLen   = Int(data[8]) | (Int(data[9]) << 8)
            headerStart = 10
        } else {
            headerLen   = Int(data[8])  | (Int(data[9])  << 8)
                        | (Int(data[10]) << 16) | (Int(data[11]) << 24)
            headerStart = 12
        }
        dataOffset = headerStart + headerLen

        let hdr = String(
            data: data.subdata(in: headerStart ..< (headerStart + headerLen)),
            encoding: .utf8) ?? ""
        let re = try NSRegularExpression(pattern: #"'shape':\s*\((\d+),\s*(\d+)\)"#)
        let ns = hdr as NSString
        guard let m = re.firstMatch(in: hdr, range: NSRange(hdr.startIndex..., in: hdr)) else {
            throw TTSError.npyFormat("cannot parse shape in \(url.lastPathComponent)")
        }
        rows = Int(ns.substring(with: m.range(at: 1)))!
        cols = Int(ns.substring(with: m.range(at: 2)))!
    }

    func row(_ i: Int) -> [Float32] {
        let off = dataOffset + i * cols * MemoryLayout<Float32>.size
        let end = off + cols * MemoryLayout<Float32>.size
        return data.subdata(in: off ..< end).withUnsafeBytes { buf in
            Array(buf.bindMemory(to: Float32.self))
        }
    }
}

// MARK: - Errors

enum TTSError: LocalizedError {
    case notLoaded
    case tooShort
    case noFrames
    case npyFormat(String)
    case mlModel(String)
    var errorDescription: String? {
        switch self {
        case .notLoaded:        return "Models not loaded yet"
        case .tooShort:         return "Text too short"
        case .noFrames:         return "No audio frames generated"
        case .npyFormat(let s): return "NPY error: \(s)"
        case .mlModel(let s):   return "CoreML error: \(s)"
        }
    }
}

// MARK: - InferenceState
// Holds all inference resources; marked @unchecked Sendable so it can be
// captured in a Task.detached closure without triggering sendability errors.

final class InferenceState: @unchecked Sendable {
    var decode:        MLModel?
    var codePredictor: MLModel?
    var vocoder:       MLModel?
    var codecEmbed:    [[Float32]] = []       // [3072, 1024]
    var cpEmbeds:      [[[Float32]]] = []     // 15 × [2048, 1024]
    var textReader:    NpyRowReader?
    var tokenizer:     (any Tokenizer)?

    // MARK: Embedding helpers

    private func textRow(_ id: Int) -> [Float32] { textReader!.row(id) }

    private func vadd(_ a: [Float32], _ b: [Float32]) -> [Float32] {
        var r = a; for i in 0..<kHidden { r[i] += b[i] }; return r
    }

    // MARK: buildInputEmbeds
    // Non-streaming, English, no speaker → [8 + textLen + 2, kHidden]
    //
    // Python non_streaming_mode=True layout (modeling_qwen3_tts.py):
    //   [0..2]  role: text_projection(text_embedding([im_start, assistant, \n]))
    //   [3..7]  codec prefix: (tts_pad×4 + tts_bos) + codec_input_embedding[:-1]
    //             = [ttsPad+think, ttsPad+think_bos, ttsPad+english, ttsPad+think_eos, ttsBos+codecPad]
    //   [8..8+N-1]  text tokens + codec_pad  (N = textLen)
    //   [8+N]   tts_eos + codec_pad
    //   [8+N+1] tts_pad + codec_bos
    func buildInputEmbeds(textTokenIds: [Int]) -> [[Float32]] {
        let ttsBos = textRow(kTtsBosId)
        let ttsEos = textRow(kTtsEosId)
        let ttsPad = textRow(kTtsPadId)
        let cPad   = codecEmbed[kCPadId]
        let cBos   = codecEmbed[kCBosId]

        // role embed: 3 rows
        var embeds = kRoleTokens.map { textRow($0) }

        // codec prefix: 5 rows — (tts_pad×4 + tts_bos) + codec_input_embedding[:-1]
        // codec_input_embedding (no speaker, english) = [think, think_bos, english, think_eos, pad, bos]
        // codec_input_embedding[:-1] = [think, think_bos, english, think_eos, pad]
        let cPfx = [kCThinkId, kCTBosId, kCEnglish, kCTEosId, kCPadId].map { codecEmbed[$0] }
        for i in 0..<4 { embeds.append(vadd(ttsPad, cPfx[i])) }
        embeds.append(vadd(ttsBos, cPfx[4]))   // ttsBos + codecPad

        // text tokens each + codec_pad  → textLen rows
        for id in textTokenIds { embeds.append(vadd(textRow(id), cPad)) }
        // tts_eos + codec_pad  → 1 row
        embeds.append(vadd(ttsEos, cPad))
        // tts_pad + codec_bos  → 1 row
        embeds.append(vadd(ttsPad, cBos))

        return embeds   // shape: [8 + textLen + 2, kHidden]
    }

    // MARK: runDecode
    // Single-token decode step. Reads logits/hidden at `pos`, updates KV cache in-place.
    // kCache/vCache are arrays of MLMultiArray; outputs are swapped in directly (no copy).
    func runDecode(embed: [Float32], pos: Int,
                   kCache: inout [MLMultiArray],
                   vCache: inout [MLMultiArray]) async throws -> (logits: [Float32], hidden: [Float32]) {
        guard let model = decode else { throw TTSError.notLoaded }

        // inputs_embeds [1, 1, kHidden] Float16
        let eArr = try MLMultiArray(shape: [1, 1, kHidden] as [NSNumber], dataType: .float16)
        eArr.withUnsafeMutableBytes { ptr, _ in
            let p = ptr.bindMemory(to: Float16.self)
            for j in 0..<kHidden { p[j] = Float16(embed[j]) }
        }

        // causal_mask_row [1, 1, 1, kMaxLen] Float16 — 0 for kv <= pos, -inf elsewhere
        let mArr = try MLMultiArray(shape: [1, 1, 1, kMaxLen] as [NSNumber], dataType: .float16)
        mArr.withUnsafeMutableBytes { ptr, _ in
            let p = ptr.bindMemory(to: Float16.self)
            for kv in 0..<kMaxLen { p[kv] = kv <= pos ? 0 : -.infinity }
        }

        // position_ids [3, 1, 1] Int32 — all equal to pos (mRoPE)
        let posArr = try MLMultiArray(shape: [3, 1, 1] as [NSNumber], dataType: .int32)
        posArr.withUnsafeMutableBytes { ptr, _ in
            let p = ptr.bindMemory(to: Int32.self)
            p[0] = Int32(pos); p[1] = Int32(pos); p[2] = Int32(pos)
        }

        // cache_pos [1] Int32
        let cpArr = try MLMultiArray(shape: [1] as [NSNumber], dataType: .int32)
        cpArr.withUnsafeMutableBytes { ptr, _ in
            ptr.bindMemory(to: Int32.self)[0] = Int32(pos)
        }

        var dict: [String: MLFeatureValue] = [
            "inputs_embeds":   MLFeatureValue(multiArray: eArr),
            "causal_mask_row": MLFeatureValue(multiArray: mArr),
            "position_ids":    MLFeatureValue(multiArray: posArr),
            "cache_pos":       MLFeatureValue(multiArray: cpArr),
        ]
        // The Python forward takes *kv_flat interleaved: k_0,v_0,k_1,v_1,...,k_27,v_27
        // ct.convert names them SEQUENTIALLY: k_0..k_27, v_0..v_27 (56 items positionally).
        // So CoreML input named "k_m" (m<28) corresponds to kv_flat[m]:
        //   even m → kCache[m/2], odd m → vCache[m/2]
        // And "v_j" (j<28) corresponds to kv_flat[28+j]:
        //   even (28+j) → kCache[(28+j)/2], odd (28+j) → vCache[(28+j)/2]
        for m in 0..<kNumLayers {
            let layer = m / 2
            dict["k_\(m)"] = MLFeatureValue(multiArray: m % 2 == 0 ? kCache[layer] : vCache[layer])
        }
        for j in 0..<kNumLayers {
            let m = kNumLayers + j
            let layer = m / 2
            dict["v_\(j)"] = MLFeatureValue(multiArray: m % 2 == 0 ? kCache[layer] : vCache[layer])
        }

        let inp = try MLDictionaryFeatureProvider(dictionary: dict)
        let out = try await model.prediction(from: inp)

        guard let logitsMA = out.featureValue(for: "linear_196")?.multiArrayValue,
              let hiddenMA = out.featureValue(for: "_to_copy_398")?.multiArrayValue
        else { throw TTSError.mlModel("missing decode outputs") }

        if pos == 0 {
            let embedNorm = sqrt(embed.map { $0*$0 }.reduce(0, +))
            print("[TTS] pos=0 embed norm=\(embedNorm)")
            print("[TTS] output feature names: \(out.featureNames.sorted())")
            print("[TTS] logitsMA: shape=\(logitsMA.shape) dtype=\(logitsMA.dataType.rawValue)")
            print("[TTS] logits[0..3]: \(logitsMA[0].floatValue) \(logitsMA[1].floatValue) \(logitsMA[2].floatValue) \(logitsMA[3].floatValue)")
            print("[TTS] hiddenMA: shape=\(hiddenMA.shape) dtype=\(hiddenMA.dataType.rawValue)")
            print("[TTS] hidden[0..3]: \(hiddenMA[0].floatValue) \(hiddenMA[1].floatValue) \(hiddenMA[2].floatValue) \(hiddenMA[3].floatValue)")
        }

        func toFloat32Array(_ ma: MLMultiArray) -> [Float32] {
            let n = ma.count
            switch ma.dataType {
            case .float16:
                let p = ma.dataPointer.bindMemory(to: Float16.self, capacity: n)
                return (0..<n).map { Float(p[$0]) }
            case .float32:
                let p = ma.dataPointer.bindMemory(to: Float32.self, capacity: n)
                return (0..<n).map { p[$0] }
            case .double:
                let p = ma.dataPointer.bindMemory(to: Double.self, capacity: n)
                return (0..<n).map { Float($0) }
            default:
                return []
            }
        }

        let logits = toFloat32Array(logitsMA)
        let hidden = toFloat32Array(hiddenMA)

        // Swap updated KV tensors by reference — CoreML allocates new tensors for scatter outputs,
        // and MLMultiArray is ARC-managed, so kCache/vCache will keep them alive.
        // Outputs are interleaved: scatter=k_0, scatter_1=v_0, scatter_2=k_1, scatter_3=v_1, ...
        // First output is named "scatter" (no _0), rest follow "scatter_N" numbering.
        for i in 0..<kNumLayers {
            let kName = (i == 0) ? "scatter" : "scatter_\(2 * i)"
            let vName = "scatter_\(2 * i + 1)"
            if let ma = out.featureValue(for: kName)?.multiArrayValue { kCache[i] = ma }
            if let ma = out.featureValue(for: vName)?.multiArrayValue { vCache[i] = ma }
        }
        // Verify cache was updated on first step
        if pos == 0 {
            let updated = out.featureValue(for: "scatter") != nil
            print("[TTS] KV cache update at pos=0: \(updated ? "OK" : "FAILED - scatter not found in output")")
        }

        return (logits, hidden)
    }

    // MARK: runCodePredictor
    func runCodePredictor(lastHidden: [Float32], cb0Embed: [Float32]) async throws -> [Int] {
        guard let model = codePredictor else { throw TTSError.notLoaded }
        var seq: [[Float32]] = [lastHidden, cb0Embed]
        var result: [Int] = []

        for k in 0..<15 {
            let seqLen = seq.count
            let pad    = kMaxCpLen - seqLen

            let eArr = try MLMultiArray(
                shape: [1, kMaxCpLen, kHidden] as [NSNumber], dataType: .float32)
            eArr.withUnsafeMutableBytes { ptr, _ in
                let p = ptr.bindMemory(to: Float32.self)
                for i in 0..<p.count { p[i] = 0 }
                for (i, row) in seq.enumerated() {
                    let off = (pad + i) * kHidden
                    for j in 0..<kHidden { p[off + j] = row[j] }
                }
            }

            let mArr = try MLMultiArray(
                shape: [1, 1, kMaxCpLen, kMaxCpLen] as [NSNumber], dataType: .float32)
            mArr.withUnsafeMutableBytes { ptr, _ in
                let p = ptr.bindMemory(to: Float32.self)
                for i in 0..<p.count { p[i] = -.infinity }
                for i in 0..<pad { p[i * kMaxCpLen + i] = 0 }
                for i in 0..<seqLen {
                    let row = pad + i
                    for j in pad...(pad + i) { p[row * kMaxCpLen + j] = 0 }
                }
            }

            let inp = try MLDictionaryFeatureProvider(dictionary: [
                "inputs_embeds":  MLFeatureValue(multiArray: eArr),
                "attention_mask": MLFeatureValue(multiArray: mArr),
            ])
            let out = try await model.prediction(from: inp)
            guard let logitsMA = out.featureValue(for: "all_logits")?.multiArrayValue
            else { throw TTSError.mlModel("missing all_logits") }

            let n = logitsMA.count
            let lp = logitsMA.dataPointer.bindMemory(to: Float32.self, capacity: n)
            let cbLogits = (k * kCodecVocab ..< (k + 1) * kCodecVocab).map { lp[$0] }

            let next = sampleToken(logits: cbLogits, doSample: kSubSample,
                                   temperature: kSubTemp, topK: kSubTopK)
            result.append(next)
            if k < 14 { seq.append(cpEmbeds[k][next]) }
        }
        return result
    }

    // MARK: runVocoder
    func runVocoder(frames: [[Int]]) async throws -> [Float32] {
        guard let model = vocoder else { throw TTSError.notLoaded }
        let N = frames.count
        var output: [Float32] = []
        var start = 0

        while start < N {
            let end     = min(start + kCodecChunk, N)
            let context = min(kCodecContext, start)
            let chunkSrc = start - context

            let cArr = try MLMultiArray(
                shape: [1, kNumStreams, kCodecWindow] as [NSNumber], dataType: .int32)
            cArr.withUnsafeMutableBytes { ptr, _ in
                let p = ptr.bindMemory(to: Int32.self)
                for i in 0..<p.count { p[i] = 0 }
                let chunkLen = end - chunkSrc
                for f in 0..<min(chunkLen, kCodecWindow) {
                    let src = chunkSrc + f
                    guard src >= 0 && src < N else { continue }
                    for s in 0..<kNumStreams { p[s * kCodecWindow + f] = Int32(frames[src][s]) }
                }
            }

            let inp = try MLDictionaryFeatureProvider(dictionary: [
                "codes": MLFeatureValue(multiArray: cArr)
            ])
            let out  = try await model.prediction(from: inp)
            guard let wavMA = out.featureValue(for: "waveform")?.multiArrayValue
            else { throw TTSError.mlModel("missing waveform") }

            let discard = context * kUpsample
            let keep    = (end - start) * kUpsample
            let wavN    = wavMA.count
            let wavP    = wavMA.dataPointer.bindMemory(to: Float32.self, capacity: wavN)
            for i in discard ..< (discard + keep) where i < wavN { output.append(wavP[i]) }
            start = end
        }
        return output
    }

    // MARK: sampleToken
    func sampleToken(logits: [Float32], doSample: Bool,
                     temperature: Float, topK: Int) -> Int {
        guard !logits.isEmpty else { return 0 }
        if !doSample { return logits.indices.max(by: { logits[$0] < logits[$1] }) ?? 0 }
        var s = logits.map { $0 / temperature }
        if topK > 0 && topK < s.count {
            let thr = s.sorted(by: >)[topK - 1]
            for i in s.indices where s[i] < thr { s[i] = -.infinity }
        }
        let mx = s.max() ?? 0
        var p  = s.map { expf($0 - mx) }
        let sm = p.reduce(0, +)
        guard sm > 0 else { return 0 }
        for i in p.indices { p[i] /= sm }
        var cum: Float = 0
        let r = Float.random(in: 0..<1)
        for i in p.indices { cum += p[i]; if r < cum { return i } }
        return p.indices.last ?? 0
    }

    // MARK: Main pipeline
    func runPipeline(text: String,
                     onProgress: @escaping (String) -> Void) async throws -> Data {
        guard let tok = tokenizer else { throw TTSError.notLoaded }

        onProgress("Tokenizing…")
        let prompt  = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let allIds  = tok.encode(text: prompt, addSpecialTokens: false)
        guard allIds.count > 8 else { throw TTSError.tooShort }
        let textIds = Array(allIds[3 ..< (allIds.count - 5)])

        let embeds = buildInputEmbeds(textTokenIds: textIds)
        let T      = embeds.count
        let ttsPad = textRow(kTtsPadId)

        // In non_streaming_mode=True (what the export uses), trailing_text_hidden = tts_pad_embed (constant).
        // The full text is already in the prefill embeds; only a constant pad is added each gen step.
        // (Streaming mode would use per-step text embeds, but we don't use that path.)

        // Allocate KV cache: 28 K + 28 V arrays, each [1, 8, 2048, 128] Float16, zeroed
        let kvShape = [1, kNumKVHeads, kMaxLen, kHeadDim] as [NSNumber]
        var kCache: [MLMultiArray] = try (0..<kNumLayers).map { _ in
            let a = try MLMultiArray(shape: kvShape, dataType: .float16)
            a.withUnsafeMutableBytes { ptr, _ in
                ptr.bindMemory(to: Float16.self).initialize(repeating: 0)
            }
            return a
        }
        var vCache: [MLMultiArray] = try (0..<kNumLayers).map { _ in
            let a = try MLMultiArray(shape: kvShape, dataType: .float16)
            a.withUnsafeMutableBytes { ptr, _ in
                ptr.bindMemory(to: Float16.self).initialize(repeating: 0)
            }
            return a
        }

        // Prefill phase: process initial context token-by-token to warm KV cache
        onProgress("Prefilling context…")
        var lastLogits: [Float32] = []
        var lastHidden: [Float32] = []
        for pos in 0..<T {
            if pos % 8 == 0 { onProgress("Prefilling… (\(pos)/\(T))") }
            (lastLogits, lastHidden) = try await runDecode(
                embed: embeds[pos], pos: pos, kCache: &kCache, vCache: &vCache)
        }

        // Diagnostics: check prefill logits look sane
        let topLogits = lastLogits.enumerated().sorted { $0.element > $1.element }.prefix(5)
        print("[TTS] Post-prefill top-5 logits: \(topLogits.map { "[\($0.offset)]=\(String(format: "%.2f", $0.element))" })")
        let validMass = lastLogits.prefix(kCodecVocab).map { expf($0) }.reduce(0, +)
        let totalMass = lastLogits.map { expf($0) }.reduce(0, +)
        print(String(format: "[TTS] Valid codec token mass: %.1f%%", 100 * validMass / totalMass))

        // Generate phase: autoregressively decode codec frames
        var frames:    [[Int]] = []
        let maxFrames = 1500
        var pos       = T
        var outOfRangeCount = 0

        let minFrames = 12   // require at least ~1s before allowing EOS
        func sampleCb0(from logits: [Float32], allowEos: Bool) -> Int {
            if allowEos { return sampleToken(logits: logits, doSample: kDoSample, temperature: kTemp, topK: kTopK) }
            var masked = logits; masked[kEosId] = -.infinity
            return sampleToken(logits: masked, doSample: kDoSample, temperature: kTemp, topK: kTopK)
        }
        var cb0 = sampleCb0(from: lastLogits, allowEos: false)

        for step in 0..<maxFrames {
            if step % 24 == 0 { onProgress("Generating… (\(step) frames)") }
            guard pos < kMaxLen else { break }
            if cb0 == kEosId && step >= minFrames { break }
            if cb0 == kEosId { cb0 = sampleCb0(from: lastLogits, allowEos: false) }

            let cb0Embed = codecEmbed[cb0]
            let cbRest   = try await runCodePredictor(lastHidden: lastHidden, cb0Embed: cb0Embed)
            frames.append([cb0] + cbRest)

            // Build next embed: cb0_embed + sum(cp_embs[k][cbRest[k]]) + tts_pad
            // Matches generate_with_coreml (the working prefill loop).
            var next = [Float32](repeating: 0, count: kHidden)
            for j in 0..<kHidden { next[j] = cb0Embed[j] + ttsPad[j] }
            for k in 0..<15 {
                let emb = cpEmbeds[k][cbRest[k]]
                for j in 0..<kHidden { next[j] += emb[j] }
            }

            if cb0 >= kCodecVocab { outOfRangeCount += 1 }

            let t0 = Date()
            (lastLogits, lastHidden) = try await runDecode(
                embed: next, pos: pos, kCache: &kCache, vCache: &vCache)
            let elapsed = Date().timeIntervalSince(t0)
            if step < 3 { print(String(format: "[TTS] gen step %d decode: %.3fs  cb0=%d", step, elapsed, cb0)) }
            pos += 1

            cb0 = sampleToken(logits: lastLogits, doSample: kDoSample,
                              temperature: kTemp, topK: kTopK)
        }

        guard !frames.isEmpty else { throw TTSError.noFrames }

        let cb0s = frames.prefix(10).map { $0[0] }
        print("[TTS] Generated \(frames.count) frames. First 10 cb0: \(cb0s)")
        if let first = frames.first {
            print("[TTS] First frame all 16 codes: \(first)")
        }
        print("[TTS] Out-of-range cb0 (>=\(kCodecVocab)): \(outOfRangeCount) / \(frames.count)")

        onProgress("Decoding \(frames.count) frames…")
        let samples = try await runVocoder(frames: frames)

        let vMin = samples.min() ?? 0
        let vMax = samples.max() ?? 0
        let vMean = samples.reduce(0, +) / Float(samples.count)
        let vRms = sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(samples.count))
        print("[TTS] Waveform: \(samples.count) samples, min=\(vMin), max=\(vMax), mean=\(vMean), rms=\(vRms)")

        return makeWav(samples: samples)
    }

    // MARK: WAV builder
    private func makeWav(samples: [Float32]) -> Data {
        // Peak-normalize so output is always audible (target peak = 0.9)
        let peak = samples.map { abs($0) }.max() ?? 0
        let scale: Float32 = peak > 0.001 ? 0.9 / peak : 1.0
        let clip = samples.map { max(-1.0, min(1.0, $0 * scale)) }
        let n    = clip.count
        var d    = Data(capacity: 44 + n * 2)

        func u32(_ v: UInt32) -> [UInt8] { withUnsafeBytes(of: v.littleEndian) { Array($0) } }
        func u16(_ v: UInt16) -> [UInt8] { withUnsafeBytes(of: v.littleEndian) { Array($0) } }

        d += "RIFF".utf8;    d += u32(UInt32(36 + n * 2))
        d += "WAVE".utf8
        d += "fmt ".utf8;    d += u32(16)
        d += u16(1);         d += u16(1)               // PCM, mono
        d += u32(UInt32(kSampleRate))
        d += u32(UInt32(kSampleRate * 2))               // byte rate
        d += u16(2);         d += u16(16)               // block align, bits
        d += "data".utf8;    d += u32(UInt32(n * 2))
        for s in clip {
            d += u16(UInt16(bitPattern: Int16((s * 32767).rounded())))
        }
        return d
    }
}

// MARK: - Qwen3TTSEngine (MainActor)

@MainActor
final class Qwen3TTSEngine: NSObject, ObservableObject, AVAudioPlayerDelegate {

    // MARK: Published state
    @Published var isSpeaking    = false
    @Published var isLoading     = false
    @Published var statusMessage = "Ready"

    // MARK: Private
    private var synthesisTask: Task<Void, Never>?
    private var audioPlayer:   AVAudioPlayer?
    private let state = InferenceState()

    override init() {
        super.init()
    }

    // MARK: Public API

    func speak(text: String) {
        stop()
        synthesisTask = Task { await synthesize(text: text) }
    }

    func stop() {
        synthesisTask?.cancel()
        synthesisTask = nil
        audioPlayer?.stop()
        audioPlayer = nil
        isSpeaking    = false
        statusMessage = "Ready"
    }

    // MARK: AVAudioPlayerDelegate

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer,
                                                 successfully _: Bool) {
        Task { @MainActor in self.isSpeaking = false; self.statusMessage = "Ready" }
    }

    nonisolated func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer,
                                                     error: Error?) {
        Task { @MainActor in
            self.isSpeaking    = false
            self.statusMessage = "Error: \(error?.localizedDescription ?? "decode failed")"
        }
    }

    // MARK: Asset loading

    private func loadAssets() async {
        isLoading     = true
        statusMessage = "Loading…"
        do {
            try await Task.detached(priority: .userInitiated) { [state, weak self] in
                try await state.loadAll { msg in
                    Task { @MainActor [weak self] in self?.statusMessage = msg }
                }
            }.value
            statusMessage = "Ready"
        } catch {
            statusMessage = "Load error: \(error.localizedDescription)"
            print("[Qwen3TTS] Load error: \(error)")
        }
        isLoading = false
    }

    // MARK: Synthesis

    private func synthesize(text: String) async {
        if state.decode == nil {
            await loadAssets()
            guard state.decode != nil else { return }
        }
        isLoading     = true
        isSpeaking    = false
        statusMessage = "Starting…"

        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .default)
        try? session.setActive(true)
        #endif

        do {
            let capturedState = state
            let wavData = try await Task.detached(priority: .userInitiated) { [weak self, capturedState] in
                try await capturedState.runPipeline(text: text) { msg in
                    Task { @MainActor [weak self] in self?.statusMessage = msg }
                }
            }.value

            try Task.checkCancellation()

            isLoading     = false
            statusMessage = "Speaking…"
            isSpeaking    = true

            let player = try AVAudioPlayer(data: wavData)
            player.delegate = self
            audioPlayer = player
            player.play()

        } catch is CancellationError {
            isLoading = false; isSpeaking = false; statusMessage = "Ready"
        } catch {
            isLoading = false; isSpeaking = false
            statusMessage = "Error: \(error.localizedDescription)"
            print("[Qwen3TTS] Synthesis error: \(error)")
        }
    }
}

// MARK: - InferenceState: asset loading

extension InferenceState {
    func loadAll(onProgress: @Sendable (String) -> Void = { _ in }) async throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuOnly
        let b = Bundle.main

        // Derive project paths from this source file's compile-time location.
        // TTSEngine.swift: <project>/demo_ttl/demo_ttl/TTSEngine.swift
        let srcDir      = URL(fileURLWithPath: #file).deletingLastPathComponent()
        let projectDir  = srcDir.deletingLastPathComponent()
        let pipelineDir = projectDir.appendingPathComponent("qwen3_tts_test_pipeline")

        // CoreML models: try bundle first (mlmodelc preferred), then filesystem
        // (mlmodelc preferred over mlpackage to skip runtime compilation).
        func mlpkg(_ name: String) throws -> MLModel {
            if let u = b.url(forResource: name, withExtension: "mlmodelc") {
                return try MLModel(contentsOf: u, configuration: cfg)
            }
            if let u = b.url(forResource: name, withExtension: "mlpackage") {
                return try MLModel(contentsOf: u, configuration: cfg)
            }
            let compiledURL = pipelineDir.appendingPathComponent("\(name).mlmodelc")
            if FileManager.default.fileExists(atPath: compiledURL.path) {
                return try MLModel(contentsOf: compiledURL, configuration: cfg)
            }
            let fsURL = pipelineDir.appendingPathComponent("\(name).mlpackage")
            guard FileManager.default.fileExists(atPath: fsURL.path)
            else { throw TTSError.mlModel("\(name) not found in bundle or at \(fsURL.path)") }
            return try MLModel(contentsOf: fsURL, configuration: cfg)
        }

        onProgress("Loading talker…")
        decode        = try mlpkg("talker_decode")
        onProgress("Loading code predictor…")
        codePredictor = try mlpkg("code_predictor")
        onProgress("Loading vocoder…")
        vocoder       = try mlpkg("speech_tokenizer_decode")

        // Embedding assets: always load from project filesystem (too large to bundle).
        let assetsURL = pipelineDir.appendingPathComponent("swift_assets")
        guard FileManager.default.fileExists(atPath: assetsURL.path)
        else { throw TTSError.mlModel("swift_assets not found at \(assetsURL.path)") }

        onProgress("Mapping text embeddings…")
        textReader = try NpyRowReader(
            url: assetsURL.appendingPathComponent("text_embeddings.npy"))

        onProgress("Loading codec embeddings…")
        let cReader = try NpyRowReader(
            url: assetsURL.appendingPathComponent("codec_embedding.npy"))
        codecEmbed = (0..<cReader.rows).map { cReader.row($0) }

        cpEmbeds = try (0..<15).map { k in
            let r = try NpyRowReader(
                url: assetsURL.appendingPathComponent("cp_codec_embedding_\(k).npy"))
            return (0..<r.rows).map { r.row($0) }
        }

        onProgress("Loading tokenizer…")
        // Tokenizer: try bundle first, fall back to source directory.
        let bundleTok = (b.resourceURL ?? b.bundleURL).appendingPathComponent("tokenizer_files")
        let tokURL    = FileManager.default.fileExists(atPath: bundleTok.path)
                      ? bundleTok
                      : srcDir.appendingPathComponent("tokenizer_files")
        guard FileManager.default.fileExists(atPath: tokURL.path)
        else { throw TTSError.mlModel("tokenizer_files not found at \(tokURL.path)") }

        tokenizer = try await AutoTokenizer.from(modelFolder: tokURL)

        // Warm up: triggers hardware-specific specialization now (one-time, cached)
        onProgress("Warming up…")
        if let model = decode {
            let kvShape = [1, kNumKVHeads, kMaxLen, kHeadDim] as [NSNumber]
            var kw: [MLMultiArray] = try (0..<kNumLayers).map { _ in
                try MLMultiArray(shape: kvShape, dataType: .float16) }
            var vw: [MLMultiArray] = try (0..<kNumLayers).map { _ in
                try MLMultiArray(shape: kvShape, dataType: .float16) }
            _ = try? await runDecode(embed: [Float32](repeating: 0, count: kHidden),
                                     pos: 0, kCache: &kw, vCache: &vw)
        }
    }
}
