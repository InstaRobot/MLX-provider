import SwiftUI

struct ContentView: View {
    @EnvironmentObject var appState: AppState
    @State private var selectedContentTab = 0

    var body: some View {
        HSplitView {
            // MARK: - Left Sidebar - Model List
            ModelListSidebar()
                .frame(minWidth: 280, idealWidth: 320, maxWidth: 400)

            // MARK: - Right Content - Server Control & Logs
            VStack(spacing: 0) {
                // Server Control Panel
                ServerControlPanel()

                Divider()

                // Main Content Area
                if appState.serverStatus == .running {
                    // Tab selector
                    Picker("", selection: $selectedContentTab) {
                        Text("Server Info").tag(0)
                        Text("Test Chat").tag(1)
                    }
                    .pickerStyle(.segmented)
                    .padding(.horizontal)
                    .padding(.top, 8)

                    if selectedContentTab == 0 {
                        ServerInfoPanel()
                    } else {
                        TestChatPanel()
                    }
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
        .overlay {
            if let progress = appState.startupProgress {
                StartupProgressOverlay(progress: progress)
            }
        }
    }
}

// MARK: - Startup Progress Overlay

struct StartupProgressOverlay: View {
    let progress: StartupProgress

    var body: some View {
        ZStack {
            Color.black.opacity(0.4)
                .ignoresSafeArea()

            VStack(spacing: 24) {
                Image(systemName: "cpu.fill")
                    .font(.system(size: 48))
                    .foregroundColor(.accentColor)

                Text("MLX-provider")
                    .font(.title)
                    .fontWeight(.bold)

                VStack(alignment: .leading, spacing: 16) {
                    ProgressStepRow(
                        step: StartupStep.checkingEnvironment,
                        isActive: progress.step == .checkingEnvironment,
                        isCompleted: stepOrder(progress.step) > stepOrder(.checkingEnvironment)
                    )

                    ProgressStepRow(
                        step: StartupStep.creatingVirtualEnvironment,
                        isActive: progress.step == .creatingVirtualEnvironment,
                        isCompleted: stepOrder(progress.step) > stepOrder(.creatingVirtualEnvironment)
                    )

                    ProgressStepRow(
                        step: StartupStep.installingDependencies,
                        isActive: progress.step == .installingDependencies,
                        isCompleted: stepOrder(progress.step) > stepOrder(.installingDependencies)
                    )

                    ProgressStepRow(
                        step: StartupStep.startingServer,
                        isActive: progress.step == .startingServer,
                        isCompleted: stepOrder(progress.step) > stepOrder(.startingServer)
                    )
                }
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))
                .cornerRadius(12)

                Text(progress.message)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(40)
            .background(Color(nsColor: .windowBackgroundColor))
            .cornerRadius(16)
            .shadow(radius: 20)
        }
    }

    private func stepOrder(_ step: StartupStep) -> Int {
        switch step {
        case .checkingEnvironment: return 0
        case .creatingVirtualEnvironment: return 1
        case .installingDependencies: return 2
        case .startingServer: return 3
        }
    }
}

struct ProgressStepRow: View {
    let step: StartupStep
    let isActive: Bool
    let isCompleted: Bool

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(backgroundColor)
                    .frame(width: 28, height: 28)

                if isCompleted {
                    Image(systemName: "checkmark")
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                } else if isActive {
                    ProgressView()
                        .scaleEffect(0.7)
                        .tint(.white)
                } else {
                    Image(systemName: step.icon)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Text(step.rawValue)
                .font(.body)
                .foregroundColor(isActive ? .primary : (isCompleted ? .primary : .secondary))

            Spacer()
        }
    }

    private var backgroundColor: Color {
        if isCompleted {
            return .green
        } else if isActive {
            return .accentColor
        } else {
            return Color(nsColor: .controlBackgroundColor)
        }
    }
}

// MARK: - Model List Sidebar

struct ModelListSidebar: View {
    @EnvironmentObject var appState: AppState

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
                                isSelected: appState.selectedModelId == model.id,
                                isLoaded: appState.loadedModel == model.id
                            )
                            .onTapGesture {
                                appState.selectedModelId = model.id
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
    @EnvironmentObject var appState: AppState
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

                    if isSelected {
                        SelectedBadge()
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
                        Text(formatSize(model.size))
                            .font(.caption2)
                    }
                }
                .foregroundColor(.secondary)
            }

            Spacer()

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

// MARK: - Selected Badge

struct SelectedBadge: View {
    var body: some View {
        Text("SELECTED")
            .font(.caption2)
            .fontWeight(.medium)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Color.accentColor.opacity(0.2))
            .foregroundColor(.accentColor)
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
            ActivityLogView()
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

// MARK: - Test Chat Panel

struct TestChatPanel: View {
    @EnvironmentObject var appState: AppState
    @State private var inputText = ""
    @State private var messages: [UIChatMessage] = []
    @State private var isGenerating = false
    @State private var currentResponse = ""
    @State private var errorMessage: String?

    private var selectedModelInfo: ModelInfo? {
        guard let modelId = appState.selectedModelId else { return nil }
        return appState.models.first(where: { $0.id == modelId })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Model Status Card
            modelStatusCard

            // Chat Area
            chatArea

            // Input Area
            inputArea
        }
        .padding()
    }

    private var modelStatusCard: some View {
        GroupBox("Model Status") {
            VStack(alignment: .leading, spacing: 8) {
                if let model = selectedModelInfo {
                    HStack {
                        Circle()
                            .fill(appState.selectedModelId != nil ? Color.accentColor : Color.gray)
                            .frame(width: 10, height: 10)
                        Text(appState.selectedModelId != nil ? "SELECTED" : "NOT SELECTED")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(appState.selectedModelId != nil ? .accentColor : .gray)
                        Spacer()
                    }

                    Text(model.name)
                        .font(.headline)

                    HStack(spacing: 16) {
                        if let params = model.parameterCount {
                            Label(params, systemImage: "number")
                                .font(.caption)
                        }
                        if let quant = model.quantization {
                            Label(quant, systemImage: "chart.bar.fill")
                                .font(.caption)
                        }
                        Text(formatSize(model.size))
                            .font(.caption)
                    }
                    .foregroundColor(.secondary)

                    if let contextWindow = model.contextWindow {
                        Text("Context: \(formatTokens(contextWindow)) tokens")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                } else {
                    HStack {
                        Circle()
                            .fill(Color.gray)
                            .frame(width: 10, height: 10)
                        Text("NO MODEL LOADED")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(.gray)
                    }
                    Text("Select a model to test")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
    }

    private var chatArea: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(messages) { message in
                        chatBubble(message)
                    }
                    if isGenerating && !currentResponse.isEmpty {
                        chatBubble(UIChatMessage(id: UUID().uuidString, role: .assistant, content: currentResponse))
                    }
                }
                .padding()
            }
            .background(Color(nsColor: .textBackgroundColor))
            .cornerRadius(8)
            .onChange(of: messages.count) { _, _ in
                if let last = messages.last {
                    withAnimation {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
    }

    private func chatBubble(_ message: UIChatMessage) -> some View {
        HStack(alignment: .top, spacing: 8) {
            if message.role == .assistant {
                Image(systemName: "apple.logo")
                    .font(.caption)
                    .foregroundColor(.accentColor)
            } else {
                Image(systemName: "person.fill")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            Text(message.content)
                .font(.body)
        }
        .id(message.id)
    }

    private var inputArea: some View {
        HStack {
            TextField("Type a message...", text: $inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...3)
                .frame(minHeight: 30)
                .onSubmit {
                    sendMessage()
                }

            Button(action: sendMessage) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
            }
            .buttonStyle(.plain)
            .disabled(inputText.isEmpty || isGenerating || appState.selectedModelId == nil)
        }
        .padding()
        .background(Color(nsColor: .controlBackgroundColor))
        .cornerRadius(8)
    }

    private func sendMessage() {
        guard !inputText.isEmpty else { return }

        let userMessage = UIChatMessage(id: UUID().uuidString, role: .user, content: inputText)
        messages.append(userMessage)
        inputText = ""
        isGenerating = true
        currentResponse = ""
        errorMessage = nil

        Task {
            do {
                let mlxMessages = [ChatMessage(role: .user, content: userMessage.content)]
                let stream = appState.generateStream(messages: mlxMessages)

                for try await chunk in stream {
                    await MainActor.run {
                        currentResponse += chunk.token
                    }
                }

                await MainActor.run {
                    if !currentResponse.isEmpty {
                        messages.append(UIChatMessage(id: UUID().uuidString, role: .assistant, content: currentResponse))
                    }
                    currentResponse = ""
                    isGenerating = false
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    isGenerating = false
                }
            }
        }
    }

    private func formatTokens(_ count: Int) -> String {
        if count >= 1000 {
            return String(format: "%.0fK", Double(count) / 1000.0)
        }
        return "\(count)"
    }
}

// MARK: - ChatMessage (UI version)

struct UIChatMessage: Identifiable {
    let id: String
    let role: Role
    let content: String

    enum Role {
        case user
        case assistant
    }
}

// MARK: - Preview

#Preview {
    ContentView()
        .environmentObject(AppState())
}

// MARK: - Helper Functions

private func formatSize(_ bytes: Int64) -> String {
    let formatter = ByteCountFormatter()
    formatter.countStyle = .file
    return formatter.string(fromByteCount: bytes)
}
