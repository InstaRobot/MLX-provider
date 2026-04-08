import SwiftUI

struct ContentView: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        HSplitView {
            // Left Panel - Model List
            VStack(spacing: 0) {
                HStack {
                    Text("Models")
                        .font(.headline)
                    Spacer()
                    Button(action: { appState.scanModels() }) {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(appState.isLoading)
                }
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))
                
                Divider()
                
                if appState.isLoading {
                    ProgressView()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if appState.models.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "folder.badge.questionmark")
                            .font(.system(size: 48))
                            .foregroundColor(.secondary)
                        Text("No models found")
                            .foregroundColor(.secondary)
                        Text("Configure your models directory")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List(appState.models, selection: Binding(
                        get: { appState.loadedModel },
                        set: { _ in }
                    )) { model in
                        ModelRowView(model: model, isLoaded: appState.loadedModel == model.id)
                            .tag(model.id)
                    }
                    .listStyle(.inset(alternatesRowBackgrounds: true))
                }
            }
            .frame(minWidth: 250, idealWidth: 300, maxWidth: 400)
            
            // Right Panel - Server Control & Logs
            VStack(spacing: 0) {
                // Server Control
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Server Status")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        HStack(spacing: 6) {
                            Circle()
                                .fill(appState.serverStatus.color)
                                .frame(width: 8, height: 8)
                            Text(appState.serverStatus.displayName)
                                .font(.system(.body, design: .monospaced))
                        }
                    }
                    
                    Spacer()
                    
                    if appState.serverStatus == .running {
                        Button("Stop") {
                            Task { await appState.stopServer() }
                        }
                        .buttonStyle(.bordered)
                    } else {
                        Button("Start") {
                            Task { await appState.startServer() }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(appState.serverStatus == .starting)
                    }
                }
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))
                
                Divider()
                
                // API Info
                VStack(alignment: .leading, spacing: 8) {
                    Text("API Endpoint")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    HStack {
                        Text("http://localhost:\(appState.config.apiPort)/v1/chat/completions")
                            .font(.system(.caption, design: .monospaced))
                        Spacer()
                        Button(action: {
                            #if os(macOS)
                            let pasteboard = NSPasteboard.general
                            pasteboard.clearContents()
                            pasteboard.setString("http://localhost:\(appState.config.apiPort)/v1/chat/completions", forType: .string)
                            #endif
                        }) {
                            Image(systemName: "doc.on.doc")
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(8)
                    .background(Color(nsColor: .textBackgroundColor))
                    .cornerRadius(6)
                }
                .padding()
                
                Divider()
                
                // Model Info
                if let loaded = appState.loadedModel {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Loaded Model")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(loaded)
                            .font(.system(.body, design: .monospaced))
                    }
                    .padding()
                }
                
                Spacer()
                
                // Error message
                if let error = appState.errorMessage {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.orange)
                        Spacer()
                        Button(action: { appState.errorMessage = nil }) {
                            Image(systemName: "xmark")
                        }
                        .buttonStyle(.plain)
                    }
                    .padding()
                    .background(Color.orange.opacity(0.1))
                }
            }
            .frame(minWidth: 400)
        }
        .sheet(isPresented: $appState.showSettings) {
            SettingsView()
                .environmentObject(appState)
        }
    }
}

// MARK: - Model Row

struct ModelRowView: View {
    let model: ModelInfo
    let isLoaded: Bool
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(model.name)
                        .fontWeight(isLoaded ? .semibold : .regular)
                    if isLoaded {
                        Text("LOADED")
                            .font(.caption2)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.green.opacity(0.2))
                            .foregroundColor(.green)
                            .cornerRadius(4)
                    }
                }
                
                HStack(spacing: 12) {
                    if let params = model.parameterCount {
                        Text(params)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    if let quant = model.quantization {
                        Text(quant)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Text(formatSize(model.size))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            Spacer()
        }
        .padding(.vertical, 4)
    }
    
    private func formatSize(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
}
