import SwiftUI
import Combine

struct ContentView: View {
    @StateObject private var ttsEngine = Qwen3TTSEngine()
    @State private var inputText: String = ""
    @FocusState private var isTextFieldFocused: Bool

    var body: some View {
        ZStack {
            // Background
            Color(red: 0.06, green: 0.06, blue: 0.10)
                .ignoresSafeArea()

            VStack(spacing: 0) {
                // Header
                VStack(spacing: 6) {
                    Text("TTS DEMO")
                        .font(.system(size: 13, weight: .semibold, design: .monospaced))
                        .tracking(6)
                        .foregroundColor(Color(red: 0.4, green: 0.8, blue: 0.6))

                    Text("CoreML · Qwen3-TTS · English")
                        .font(.system(size: 11, weight: .regular, design: .monospaced))
                        .tracking(2)
                        .foregroundColor(Color.white.opacity(0.25))
                }
                .padding(.top, 56)
                .padding(.bottom, 40)

                // Waveform visualizer (decorative, animates while speaking)
                WaveformView(isActive: ttsEngine.isSpeaking)
                    .frame(height: 48)
                    .padding(.horizontal, 32)
                    .padding(.bottom, 36)

                // Text input card
                VStack(alignment: .leading, spacing: 12) {
                    Text("INPUT TEXT")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .tracking(3)
                        .foregroundColor(Color.white.opacity(0.3))

                    TextEditor(text: $inputText)
                        .focused($isTextFieldFocused)
                        .font(.system(size: 16, weight: .regular, design: .default))
                        .foregroundColor(.white)
                        .scrollContentBackground(.hidden)
                        .background(Color.clear)
                        .frame(minHeight: 120, maxHeight: 200)
                        .overlay(
                            Group {
                                if inputText.isEmpty {
                                    Text("Type something to speak…")
                                        .font(.system(size: 16))
                                        .foregroundColor(Color.white.opacity(0.2))
                                        .allowsHitTesting(false)
                                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                                        .padding(.top, 8)
                                        .padding(.leading, 4)
                                }
                            }
                        )

                    Divider()
                        .background(Color(red: 0.4, green: 0.8, blue: 0.6).opacity(0.3))
                }
                .padding(20)
                .background(
                    RoundedRectangle(cornerRadius: 14)
                        .fill(Color.white.opacity(0.05))
                        .overlay(
                            RoundedRectangle(cornerRadius: 14)
                                .stroke(Color.white.opacity(0.08), lineWidth: 1)
                        )
                )
                .padding(.horizontal, 24)

                Spacer().frame(height: 20)

                // Status message
                Text(ttsEngine.statusMessage)
                    .font(.system(size: 12, weight: .regular, design: .monospaced))
                    .foregroundColor(statusColor)
                    .frame(height: 20)
                    .animation(.easeInOut(duration: 0.2), value: ttsEngine.statusMessage)

                Spacer().frame(height: 20)

                // Speak / Stop button
                Button(action: handleButtonTap) {
                    HStack(spacing: 10) {
                        Image(systemName: ttsEngine.isSpeaking ? "stop.fill" : "waveform")
                            .font(.system(size: 16, weight: .semibold))
                        Text(ttsEngine.isSpeaking ? "STOP" : "SPEAK")
                            .font(.system(size: 14, weight: .bold, design: .monospaced))
                            .tracking(3)
                    }
                    .foregroundColor(
                        ttsEngine.isSpeaking
                            ? Color(red: 1.0, green: 0.4, blue: 0.4)
                            : Color(red: 0.06, green: 0.06, blue: 0.10)
                    )
                    .frame(maxWidth: .infinity)
                    .frame(height: 54)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(
                                ttsEngine.isSpeaking
                                    ? Color(red: 1.0, green: 0.4, blue: 0.4).opacity(0.15)
                                    : Color(red: 0.4, green: 0.8, blue: 0.6)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 12)
                                    .stroke(
                                        ttsEngine.isSpeaking
                                            ? Color(red: 1.0, green: 0.4, blue: 0.4).opacity(0.6)
                                            : Color.clear,
                                        lineWidth: 1.5
                                    )
                            )
                    )
                }
                .padding(.horizontal, 24)
                .disabled(ttsEngine.isLoading || (!ttsEngine.isSpeaking && inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty))
                .opacity((ttsEngine.isLoading || (!ttsEngine.isSpeaking && inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)) ? 0.4 : 1.0)
                .animation(.easeInOut(duration: 0.15), value: ttsEngine.isSpeaking)

                Spacer()
            }
        }
        .onTapGesture {
            isTextFieldFocused = false
        }
    }

    private var statusColor: Color {
        switch ttsEngine.statusMessage {
        case let s where s.contains("Error") || s.contains("error"):
            return Color(red: 1.0, green: 0.4, blue: 0.4)
        case let s where s.contains("Speaking") || s.contains("Generating"):
            return Color(red: 0.4, green: 0.8, blue: 0.6)
        default:
            return Color.white.opacity(0.3)
        }
    }

    private func handleButtonTap() {
        isTextFieldFocused = false
        if ttsEngine.isSpeaking {
            ttsEngine.stop()
        } else {
            let trimmed = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return }
            ttsEngine.speak(text: trimmed)
        }
    }
}

// MARK: - Waveform View

struct WaveformView: View {
    let isActive: Bool
    @State private var animationPhase: Double = 0

    private let barCount = 28
    private let timer = Timer.publish(every: 0.05, on: .main, in: .common).autoconnect()

    var body: some View {
        HStack(spacing: 3) {
            ForEach(0..<barCount, id: \.self) { index in
                RoundedRectangle(cornerRadius: 2)
                    .fill(Color(red: 0.4, green: 0.8, blue: 0.6))
                    .frame(width: 3, height: barHeight(for: index))
                    .opacity(isActive ? 0.85 : 0.2)
            }
        }
        .onReceive(timer) { _ in
            if isActive {
                withAnimation(.linear(duration: 0.05)) {
                    animationPhase += 0.18
                }
            }
        }
    }

    private func barHeight(for index: Int) -> CGFloat {
        guard isActive else { return 6 }
        let base = 8.0
        let amplitude = 18.0
        let frequency = 2.2
        let phase = animationPhase + Double(index) * 0.45
        let value = sin(phase * frequency) * amplitude + base
        return CGFloat(max(4, value))
    }
}

#Preview {
    ContentView()
}
