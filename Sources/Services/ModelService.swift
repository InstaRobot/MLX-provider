import Foundation

actor ModelService {
    private var loadedModel: String?
    private var process: Process?
    private var port: Int = 8080
    private let mlxLMPath: String
    
    init() {
        // Try to find mlx_lm server script
        if let bundlePath = Bundle.main.resourcePath {
            let resourceURL = URL(fileURLWithPath: bundlePath)
            let serverPath = resourceURL.appendingPathComponent("mlx_server.py")
            if FileManager.default.fileExists(atPath: serverPath.path) {
                mlxLMPath = serverPath.path
            } else {
                mlxLMPath = "/usr/local/bin/mlx_lm_server.py"
            }
        } else {
            mlxLMPath = "/usr/local/bin/mlx_lm_server.py"
        }
    }
    
    // MARK: - Model Scanning
    
    func scanDirectory(_ path: String) throws -> [ModelInfo] {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return []
        }
        
        var models: [ModelInfo] = []
        let scanner = DirectoryScanner()
        models = try scanner.scanForModels(at: url)
        
        return models
    }
    
    // MARK: - Model Management
    
    func loadModel(_ modelId: String) async throws {
        guard let url = URL(string: "http://localhost:\(port)") else {
            throw ModelServiceError.invalidURL
        }
        
        var request = URLRequest(url: url.appendingPathComponent("v1/models/\(modelId)/load"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let (_, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw ModelServiceError.loadFailed
        }
        
        loadedModel = modelId
    }
    
    func unloadModel(_ modelId: String) async throws {
        guard let url = URL(string: "http://localhost:\(port)") else {
            throw ModelServiceError.invalidURL
        }
        
        var request = URLRequest(url: url.appendingPathComponent("v1/models/\(modelId)/unload"))
        request.httpMethod = "POST"
        
        let (_, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw ModelServiceError.unloadFailed
        }
        
        if loadedModel == modelId {
            loadedModel = nil
        }
    }
    
    func getLoadedModel() -> String? {
        return loadedModel
    }
}

// MARK: - Directory Scanner

struct DirectoryScanner {
    func scanForModels(at url: URL) throws -> [ModelInfo] {
        var models: [ModelInfo] = []
        let fileManager = FileManager.default
        let enumerator = fileManager.enumerator(
            at: url,
            includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        )
        
        while let fileURL = enumerator?.nextObject() as? URL {
            if isModelDirectory(fileURL) {
                let modelInfo = extractModelInfo(from: fileURL)
                models.append(modelInfo)
            }
        }
        
        return models
    }
    
    private func isModelDirectory(_ url: URL) -> Bool {
        let mlxFiles = ["model.safetensors", "model.bin", "model.mlx"]
        let configFiles = ["config.json"]
        
        let hasModelFile = mlxFiles.contains { fileManager.fileExists(atPath: url.appendingPathComponent($0).path) }
        let hasConfigFile = configFiles.contains { fileManager.fileExists(atPath: url.appendingPathComponent($0).path) }
        
        return hasModelFile || hasConfigFile
    }
    
    private func extractModelInfo(from url: URL) -> ModelInfo {
        let name = url.lastPathComponent
        let path = url.path
        
        var totalSize: Int64 = 0
        let fileManager = FileManager.default
        if let enumerator = fileManager.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) {
            while let fileURL = enumerator.nextObject() as? URL {
                if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                    totalSize += Int64(size)
                }
            }
        }
        
        var parameterCount: String? = nil
        var quantization: String? = nil
        
        if let configData = fileManager.contents(atPath: url.appendingPathComponent("config.json").path),
           let config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] {
            if let hiddenSize = config["hidden_size"] as? Int,
               let numLayers = config["num_hidden_layers"] as? Int,
               let numHeads = config["num_attention_heads"] as? Int {
                let approxParams = Double(hiddenSize * numLayers * numHeads * 12) / 1e9
                parameterCount = String(format: "%.1fB", approxParams)
            }
            if let quantConfig = config["quantization_config"] as? [String: Any],
               let bits = quantConfig["bits"] as? Int {
                quantization = "\(bits)-bit"
            }
        }
        
        return ModelInfo(
            id: name,
            name: name,
            path: path,
            size: totalSize,
            parameterCount: parameterCount,
            quantization: quantization
        )
    }
}

// MARK: - Errors

enum ModelServiceError: LocalizedError {
    case invalidURL
    case loadFailed
    case unloadFailed
    case serverNotRunning
    
    var errorDescription: String? {
        switch self {
        case .invalidURL: return "Invalid server URL"
        case .loadFailed: return "Failed to load model"
        case .unloadFailed: return "Failed to unload model"
        case .serverNotRunning: return "MLX server is not running"
        }
    }
}
