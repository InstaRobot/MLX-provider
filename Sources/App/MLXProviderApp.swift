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
        .windowToolbar {
            ToolbarItem(placement: .automatic) {
                Button(action: { appState.showSettings.toggle() }) {
                    Image(systemName: "gear")
                }
            }
        }
    }
}

// MARK: - App State

@MainActor
final class AppState: ObservableObject {
    @Published var config = AppConfig()
    @Published var serverStatus: ServerStatus = .stopped
    @Published var models: [ModelInfo] = []
    @Published var loadedModel: String?
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var showSettings = false
    
    private let configStore = ConfigStore()
    private let modelService: ModelService
    private let apiService: APIService
    
    init() {
        self.modelService = ModelService()
        self.apiService = APIService()
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
        Task {
            do {
                models = try await modelService.scanDirectory(config.modelDirectory)
                isLoading = false
            } catch {
                errorMessage = error.localizedDescription
                isLoading = false
            }
        }
    }
    
    func startServer() async {
        serverStatus = .starting
        do {
            try await apiService.start(
                port: config.apiPort,
                modelService: modelService
            )
            serverStatus = .running
        } catch {
            errorMessage = error.localizedDescription
            serverStatus = .stopped
        }
    }
    
    func stopServer() async {
        await apiService.stop()
        serverStatus = .stopped
    }
}

// MARK: - AppConfig

struct AppConfig: Codable {
    var modelDirectory: String = "~/Models/mlx"._NSURL.pathString
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

// MARK: - ModelInfo

struct ModelInfo: Identifiable, Codable, Hashable {
    let id: String
    let name: String
    let path: String
    let size: Int64
    let parameterCount: String?
    let quantization: String?
    
    init(id: String, name: String, path: String, size: Int64, parameterCount: String? = nil, quantization: String? = nil) {
        self.id = id
        self.name = name
        self.path = path
        self.size = size
        self.parameterCount = parameterCount
        self.quantization = quantization
    }
}
