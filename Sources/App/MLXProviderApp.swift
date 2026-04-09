import Foundation
import SwiftUI

@main
struct MLXProviderApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
        }
        .windowStyle(.titleBar)
    }
}

// MARK: - App State

@MainActor
final class AppState: ObservableObject {
    @Published var config = AppConfig()
    @Published var serverStatus: ServerStatus = .stopped
    @Published var models: [ModelInfo] = []
    @Published var selectedModelId: String?
    @Published var loadedModel: String?
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var showSettings = false
    @Published var showAbout = false
    @Published var startupProgress: StartupProgress?

    private let configStore = ConfigStore()
    private let mlxManager = MLXModelManager()

    init() {
        loadConfig()
        scanModels()
    }

    func loadConfig() {
        config = configStore.load()
    }

    func saveConfig() {
        configStore.save(config)
    }

    func scanModels() {
        isLoading = true
        let scanner = DirectoryScanner()
        let url = URL(fileURLWithPath: (config.modelDirectory as NSString).expandingTildeInPath)
        do {
            models = try scanner.scanForModels(at: url)
            isLoading = false
        } catch {
            errorMessage = error.localizedDescription
            isLoading = false
        }
    }

    func startServer() async {
        guard let modelId = selectedModelId,
              let modelInfo = models.first(where: { $0.id == modelId }) else {
            errorMessage = "Please select a model first"
            LogManager.shared.error("Failed to start server: No model selected")
            return
        }

        serverStatus = .starting
        LogManager.shared.info("Starting server...")
        startupProgress = StartupProgress(step: .checkingEnvironment, progress: 0, message: "Loading model...")

        do {
            startupProgress = StartupProgress(step: .startingServer, progress: 0.5, message: "Loading \(modelInfo.name)...")
            try await mlxManager.loadModel(modelInfo: modelInfo)

            startupProgress = nil
            loadedModel = modelId
            serverStatus = .running
            LogManager.shared.info("Server started successfully")
        } catch {
            startupProgress = nil
            errorMessage = error.localizedDescription
            serverStatus = .stopped
            LogManager.shared.error("Server failed to start: \(error.localizedDescription)")
        }
    }

    func stopServer() async {
        LogManager.shared.info("Stopping server...")
        mlxManager.unloadModel()
        loadedModel = nil
        serverStatus = .stopped
        LogManager.shared.info("Server stopped")
    }

    func loadModel(_ modelId: String) async {
        selectedModelId = modelId
    }

    func unloadModel(_ modelId: String) async {
        selectedModelId = nil
    }

    func generate(
        messages: [ChatMessage],
        temperature: Double? = nil,
        maxTokens: Int? = nil
    ) async throws -> GenerationResult {
        return try await mlxManager.generate(messages: messages)
    }

    func generateStream(
        messages: [ChatMessage],
        temperature: Double? = nil,
        maxTokens: Int? = nil
    ) -> AsyncThrowingStream<StreamChunk, Error> {
        return mlxManager.generateStream(messages: messages)
    }
}

// MARK: - AppConfig

struct AppConfig: Codable {
    var modelDirectory: String = NSString(string: "~/Models/mlx").expandingTildeInPath
    var apiPort: Int = 8080
    var apiKey: String? = nil
    var defaultModel: String? = nil
    var maxTokens: Int = 2048
    var temperature: Double = 0.7
}

// MARK: - ServerStatus

enum ServerStatus: Equatable {
    case stopped
    case starting
    case running
    case error(String)

    var displayName: String {
        switch self {
        case .stopped: return "Stopped"
        case .starting: return "Starting..."
        case .running: return "Running"
        case .error(let msg): return "Error: \(msg)"
        }
    }

    var color: Color {
        switch self {
        case .stopped: return .gray
        case .starting: return .orange
        case .running: return .green
        case .error: return .red
        }
    }
}

// MARK: - Startup Progress

enum StartupStep: String {
    case checkingEnvironment = "Checking environment"
    case creatingVirtualEnvironment = "Creating Python virtual environment"
    case installingDependencies = "Installing MLX dependencies"
    case startingServer = "Starting server"

    var icon: String {
        switch self {
        case .checkingEnvironment: return "magnifyingglass"
        case .creatingVirtualEnvironment: return "folder.badge.gearshape"
        case .installingDependencies: return "arrow.down.circle"
        case .startingServer: return "server.rack"
        }
    }
}

struct StartupProgress {
    let step: StartupStep
    let progress: Double
    let message: String
}