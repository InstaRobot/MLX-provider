import SwiftUI

struct ContentView: View {
    @EnvironmentObject var appState: AppState
    @State private var selectedModel: String?
    
    var body: some View {
        HSplitView {
            // MARK: - Left Sidebar - Model List
            ModelListSidebar(selectedModel: $selectedModel)
                .frame(minWidth: 280, idealWidth: 320, maxWidth: 400)
            
            // MARK: - Right Content - Server Control & Logs
            VStack(spacing: 0) {
                // Server Control Panel
                ServerControlPanel()
                
                Divider()
                
                // Main Content Area
                if appState.serverStatus == .running {
                    ServerInfoPanel()
                } else {
                    WelcomePanel()
                }
                
                Spacer()
                
                // Status Bar
                StatusBar()
            }
            .frame(minWidth: 450)
        }
        .toolbar {
            ToolbarItemGroup(placement: .automatic) {
                Button(action: { appState.showSettings.toggle() }) {
                    Image(systemName: "gear")
                }
                .help("Settings")
                
                Button(action: { appState.showAbout.toggle() }) {
                    Image(systemName: "info.circle")
                }
                .help("About")
            }
        }
        .sheet(isPresented: $appState.showSettings) {
            SettingsView()
                .environmentObject(appState)
        }
        .sheet(isPresented: $appState.showAbout) {
            AboutView()
        }
    }
}

// MARK: - Model List Sidebar

struct ModelListSidebar: View {
    @EnvironmentObject var appState: AppState
    @Binding var selectedModel: String?
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Label("Models", systemImage: "cpu.fill")
                    .font(.headline)
                Spacer()
                Button(action: { appState.scanModels() }) {
                    Image(systemName: "arrow.clockwise")
                }
                .disabled(appState.isLoading)
                .help("Refresh models")
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))
            
            Divider()
            
            // Model List
            if appState.isLoading {
                ProgressView()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if appState.models.isEmpty {
                EmptyModelView()
            } else {
                ScrollView {
                    LazyVStack(spacing: 1) {
                        ForEach(appState.models) { model in
                            ModelRowView(
                                model: model,
                                isSelected: selectedModel == model.id,
                                isLoaded: appState.loadedModel == model.id
                            )
                            .onTapGesture {
                                selectedModel = model.id
                            }
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
        }
    }
}

// MARK: - Empty State View

struct EmptyModelView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "folder.badge.questionmark")
                .font(.system(size: 48))
                .foregroundColor(.secondary)
            
            Text("No Models Found")
                .font(.headline)
            
            Text("Add MLX models to your models directory.\nSee Settings to configure.")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

// MARK: - Model Row

struct ModelRowView: View {
    let model: ModelInfo
    let isSelected: Bool
    let isLoaded: Bool
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(model.name)
                        .fontWeight(isLoaded ? .semibold : .regular)
                        .lineLimit(1)
                    
                    if isLoaded {
                        LoadedBadge()
                    }
                }
                
                HStack(spacing: 12) {
                    if let params = model.parameterCount {
                        Label(params, systemImage: "number")
                            .font(.caption2)
                    }
                    
                    if let quant = model.quantization {
                        Label(quant, systemImage: "chart.bar.fill")
                            .font(.caption2)
                    }
                    
                    if model.size > 0 {
                        Text(model.formattedSize)
                            .font(.caption2)
                    }
                }
                .foregroundColor(.secondary)
            }
            
            Spacer()
            
            if model.isRemote {
                Image(systemName: "cloud.fill")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(isSelected ? Color.accentColor.opacity(0.2) : Color.clear)
        )
        .contentShape(Rectangle())
    }
}

// MARK: - Loaded Badge

struct LoadedBadge: View {
    var body: some View {
        Text("LOADED")
            .font(.caption2)
            .fontWeight(.medium)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Color.green.opacity(0.2))
            .foregroundColor(.green)
            .cornerRadius(4)
    }
}

// MARK: - Server Control Panel

struct ServerControlPanel: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Server Status")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                HStack(spacing: 8) {
                    Circle()
                        .fill(appState.serverStatus.color)
                        .frame(width: 10, height: 10)
                    Text(appState.serverStatus.displayName)
                        .font(.system(.body, design: .monospaced))
                }
            }
            
            Spacer()
            
            Button(action: {
                Task {
                    if appState.serverStatus == .running {
                        await appState.stopServer()
                    } else {
                        await appState.startServer()
                    }
                }
            }) {
                HStack {
                    Image(systemName: appState.serverStatus == .running ? "stop.fill" : "play.fill")
                    Text(appState.serverStatus == .running ? "Stop" : "Start")
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(appState.serverStatus == .starting)
        }
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
    }
}

// MARK: - Server Info Panel

struct ServerInfoPanel: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // API Endpoint
            GroupBox("API Endpoint") {
                HStack {
                    Text("http://localhost:\(appState.config.apiPort)/v1/chat/completions")
                        .font(.system(.caption, design: .monospaced))
                        .lineLimit(1)
                    
                    Spacer()
                    
                    Button(action: copyEndpoint) {
                        Image(systemName: "doc.on.doc")
                    }
                    .buttonStyle(.plain)
                    .help("Copy to clipboard")
                }
            }
            
            // Loaded Model Info
            if let loadedModel = appState.loadedModel {
                GroupBox("Loaded Model") {
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                        Text(loadedModel)
                            .font(.system(.body, design: .monospaced))
                        Spacer()
                    }
                }
            }
            
            // Log Viewer
            LogView()
        }
        .padding()
    }
    
    private func copyEndpoint() {
        #if os(macOS)
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString("http://localhost:\(appState.config.apiPort)/v1/chat/completions", forType: .string)
        #endif
    }
}

// MARK: - Welcome Panel

struct WelcomePanel: View {
    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "cpu.fill")
                .font(.system(size: 64))
                .foregroundColor(.accentColor)
            
            Text("MLX-provider")
                .font(.title)
                .fontWeight(.bold)
            
            Text("OpenAI-compatible API server for MLX models\non Apple Silicon")
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
            
            VStack(alignment: .leading, spacing: 8) {
                Label("Configure models directory in Settings", systemImage: "folder")
                Label("Start the server", systemImage: "play")
                Label("Use OpenAI SDK to connect", systemImage: "link")
            }
            .font(.caption)
            .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

// MARK: - Status Bar

struct StatusBar: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        HStack {
            if let error = appState.errorMessage {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                    Text(error)
                        .font(.caption)
                        .foregroundColor(.orange)
                        .lineLimit(1)
                }
            } else {
                Text("Ready")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            if appState.models.count > 0 {
                Text("\(appState.models.count) models available")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
        .background(Color(nsColor: .controlBackgroundColor))
    }
}

// MARK: - Preview

#Preview {
    ContentView()
        .environmentObject(AppState())
}
