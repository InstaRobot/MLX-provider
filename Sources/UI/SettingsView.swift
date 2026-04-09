import SwiftUI
import UniformTypeIdentifiers

struct SettingsView: View {
    @EnvironmentObject var appState: AppState
    @Environment(\.dismiss) var dismiss

    private var selectedModelContextWindow: Int? {
        guard let defaultModelId = appState.config.defaultModel,
              let model = appState.models.first(where: { $0.id == defaultModelId }) else {
            return nil
        }
        return model.contextWindow
    }

    var body: some View {
        TabView {
            // General Tab
            Form {
                Section {
                    HStack {
                        Text("Models Directory")
                        Spacer()
                        TextField("Path", text: $appState.config.modelDirectory)
                            .frame(width: 200)
                        Button("Browse...") {
                            browseForDirectory()
                        }
                    }
                    
                    HStack {
                        Text("API Port")
                        Spacer()
                        TextField("Port", value: $appState.config.apiPort, format: .number)
                            .frame(width: 80)
                    }
                    
                    HStack {
                        Text("API Key (optional)")
                        Spacer()
                        SecureField("None", text: Binding(
                            get: { appState.config.apiKey ?? "" },
                            set: { appState.config.apiKey = $0.isEmpty ? nil : $0 }
                        ))
                        .frame(width: 200)
                    }
                }
                
                Section("Model Defaults") {
                    HStack {
                        Text("Default Model")
                        Spacer()
                        Picker("", selection: Binding(
                            get: { appState.config.defaultModel ?? "" },
                            set: { appState.config.defaultModel = $0.isEmpty ? nil : $0 }
                        )) {
                            Text("None").tag("")
                            ForEach(appState.models) { model in
                                Text(model.name).tag(model.id)
                            }
                        }
                        .frame(width: 200)
                    }
                    
                    HStack {
                        Text("Max Tokens")
                        Spacer()
                        if let contextWindow = selectedModelContextWindow {
                            Stepper(
                                "\(appState.config.maxTokens)",
                                value: $appState.config.maxTokens,
                                in: 0...max(contextWindow, appState.config.maxTokens)
                            )
                            .frame(width: 100)
                            Text("/ \(contextWindow)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        } else {
                            TextField("", value: $appState.config.maxTokens, format: .number)
                                .frame(width: 80)
                        }
                    }
                    
                    HStack {
                        Text("Temperature")
                        Spacer()
                        Slider(value: $appState.config.temperature, in: 0...2, step: 0.1)
                            .frame(width: 120)
                        Text(String(format: "%.1f", appState.config.temperature))
                            .font(.system(.body, design: .monospaced))
                            .frame(width: 40)
                    }
                }
            }
            .formStyle(.grouped)
            .tabItem {
                Label("General", systemImage: "gear")
            }
            
            // About Tab
            VStack(spacing: 16) {
                Image(systemName: "cpu.fill")
                    .font(.system(size: 64))
                    .foregroundColor(.accentColor)
                
                Text("MLX-provider")
                    .font(.title)
                    .fontWeight(.bold)
                
                Text("Version 1.0.0")
                    .foregroundColor(.secondary)
                
                Text("OpenAI-compatible API server for MLX models on Apple Silicon.")
                    .multilineTextAlignment(.center)
                    .foregroundColor(.secondary)
                
                Divider()
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("Requirements")
                        .fontWeight(.semibold)
                    Text("• Apple Silicon Mac (M1/M2/M3/M4)")
                    Text("• macOS 15.0+")
                    Text("• MLX framework")
                    Text("• Python 3.10+ (for mlx_lm backend)")
                }
                .font(.caption)
                .foregroundColor(.secondary)
                
                Spacer()
            }
            .padding()
            .tabItem {
                Label("About", systemImage: "info.circle")
            }
        }
        .frame(width: 500, height: 400)
        .onDisappear {
            appState.saveConfig()
        }
    }
    
    private func browseForDirectory() {
        #if os(macOS)
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "Select Models Directory"
        
        if panel.runModal() == .OK, let url = panel.url {
            appState.config.modelDirectory = url.path
            appState.scanModels()
        }
        #endif
    }
}
