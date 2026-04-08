import Foundation
import MLXLLM
import MLXLMCommon
import Tokenizers

// MARK: - MLX Backend Service

/// Pure Swift MLX backend for local inference - no Python required
actor MLXBackend {
    private var loadedModel: ModelContainer?
    private var loadedModelId: String?
    
    init() {}
    
    // MARK: - Model Discovery
    
    /// Scan directory for local MLX models
    func scanDirectory(_ path: String) async throws -> [ModelInfo] {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return []
        }
        
        var models: [ModelInfo] = []
        
        let contents = try FileManager.default.contentsOfDirectory(
            at: url,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )
        
        for itemURL in contents {
            var isDir: ObjCBool = false
            FileManager.default.fileExists(atPath: itemURL.path, isDirectory: &isDir)
            
            if isDir.boolValue && isModelDirectory(itemURL) {
                let modelInfo = extractModelInfo(from: itemURL)
                models.append(modelInfo)
            }
        }
        
        return models
    }
    
    private func isModelDirectory(_ url: URL) -> Bool {
        // MLX models typically have config.json and model files
        let configURL = url.appendingPathComponent("config.json")
        let hasConfig = FileManager.default.fileExists(atPath: configURL.path)
        
        // Check for common MLX model files
        let modelExtensions = ["safetensors", "bin", "mlpack", "mlir"]
        let hasModelFile = modelExtensions.contains { ext in
            let modelURL = url.appendingPathComponent("model.\(ext)")
            return FileManager.default.fileExists(atPath: modelURL.path)
        }
        
        return hasConfig || hasModelFile
    }
    
    private func extractModelInfo(from url: URL) -> ModelInfo {
        let name = url.lastPathComponent
        let path = url.path
        
        // Calculate total size
        var totalSize: Int64 = 0
        if let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) {
            while let fileURL = enumerator.nextObject() as? URL {
                if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                    totalSize += Int64(size)
                }
            }
        }
        
        // Parse config.json for metadata
        var parameterCount: String? = nil
        var quantization: String? = nil
        
        let configURL = url.appendingPathComponent("config.json")
        if let configData = FileManager.default.contents(atPath: configURL.path),
           let config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] {
            
            if let hiddenSize = config["hidden_size"] as? Int,
               let numLayers = config["num_hidden_layers"] as? Int {
                let approx = Double(hiddenSize * hiddenSize * numLayers * 12) / 1e9
                parameterCount = approx > 1 ? String(format: "%.0fB", approx) : String(format: "%.0fM", approx * 1000)
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
            quantization: quantization,
            isRemote: false
        )
    }
    
    /// List available local models (returns empty - use scanDirectory)
    func listModels() async -> [ModelInfo] {
        return []
    }
    
    // MARK: - Model Loading
    
    /// Load model from local directory
    func loadModel(path: String) async throws -> Bool {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MLXError.modelNotFound
        }
        
        guard isModelDirectory(url) else {
            throw MLXError.loadFailed("Not a valid MLX model directory")
        }
        
        do {
            print("[MLXBackend] Loading model from: \(path)")
            
            let container = try await loadModelContainer(
                from: url,
                using: TokenizersLoader()
            )
            
            loadedModel = container
            loadedModelId = url.lastPathComponent
            
            print("[MLXBackend] Successfully loaded: \(loadedModelId ?? "")")
            return true
        } catch {
            print("[MLXBackend] Failed to load model: \(error)")
            throw MLXError.loadFailed(error.localizedDescription)
        }
    }
    
    /// Load model by ID (scans default directories)
    func loadModel(id: String) async throws -> Bool {
        // For local-only, ID is the path
        return try await loadModel(path: id)
    }
    
    func unloadModel(id: String) -> Bool {
        if loadedModelId == id || loadedModelId == nil {
            loadedModel = nil
            loadedModelId = nil
            print("[MLXBackend] Model unloaded")
            return true
        }
        return false
    }
    
    func isModelLoaded(_ id: String) -> Bool {
        return loadedModel != nil && loadedModelId == id
    }
    
    func getLoadedModelId() -> String? {
        return loadedModelId
    }
    
    // MARK: - Generation
    
    func generate(
        modelId: String,
        messages: [ChatMessage],
        temperature: Double = 0.7,
        maxTokens: Int = 512,
        repetitionPenalty: Double = 1.0
    ) async throws -> GenerationResult {
        guard let container = loadedModel else {
            throw MLXError.modelNotLoaded
        }
        
        let prompt = buildPrompt(messages: messages)
        let inputIds = container.tokenizer.encode(prompt)
        let promptTokens = inputIds.count
        
        var outputIds: [Int] = []
        
        try await container.model.perform { model in
            var token = inputIds
            for _ in 0..<maxTokens {
                let logits = model(token)
                let nextToken = self.sampleToken(logits: logits, temperature: Float(temperature))
                
                if nextToken == container.tokenizer.eosTokenId {
                    break
                }
                
                outputIds.append(nextToken)
                token = [nextToken]
            }
        }
        
        let response = container.tokenizer.decode(outputIds)
        
        return GenerationResult(
            id: "chatcmpl-\(UUID().uuidString.prefix(8))",
            model: modelId,
            content: response.trimmingCharacters(in: .whitespacesAndNewlines),
            finishReason: .stop,
            promptTokens: promptTokens,
            completionTokens: outputIds.count,
            totalTokens: promptTokens + outputIds.count
        )
    }
    
    /// Streaming generation
    func generateStream(
        modelId: String,
        messages: [ChatMessage],
        temperature: Double = 0.7,
        maxTokens: Int = 512
    ) -> AsyncThrowingStream<StreamChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                guard let container = self.loadedModel else {
                    continuation.finish(throwing: MLXError.modelNotLoaded)
                    return
                }
                
                let prompt = self.buildPrompt(messages: messages)
                let inputIds = container.tokenizer.encode(prompt)
                var outputIds: [Int] = []
                
                do {
                    try await container.model.perform { model in
                        var token = inputIds
                        for _ in 0..<maxTokens {
                            let logits = model(token)
                            let nextToken = self.sampleToken(logits: logits, temperature: Float(temperature))
                            
                            if nextToken == container.tokenizer.eosTokenId {
                                break
                            }
                            
                            outputIds.append(nextToken)
                            token = [nextToken]
                            
                            let text = container.tokenizer.decode([nextToken])
                            let chunk = StreamChunk(
                                token: text,
                                tokenId: nextToken,
                                logprob: nil,
                                finishReason: nil
                            )
                            continuation.yield(chunk)
                        }
                    }
                } catch {
                    continuation.finish(throwing: error)
                    return
                }
                
                continuation.finish()
            }
        }
    }
    
    // MARK: - Helper Methods
    
    private func buildPrompt(messages: [ChatMessage]) -> String {
        var prompt = ""
        
        for message in messages {
            switch message.role {
            case .system:
                prompt += "System: \(message.content)\n\n"
            case .user:
                prompt += "User: \(message.content)\n\n"
            case .assistant:
                prompt += "Assistant: \(message.content)\n\n"
            }
        }
        
        prompt += "Assistant: "
        return prompt
    }
    
    private func sampleToken(logits: MLXArray, temperature: Float) -> Int {
        if temperature == 0 {
            return Int.argmax(logits).item(Int.self)
        }
        
        let probs = softmax(logits, axis: -1)
        return Int.argmax(probs).item(Int.self)
    }
}

// MARK: - Data Types

struct ModelInfo: Identifiable, Codable, Hashable, Sendable {
    let id: String
    let name: String
    let path: String
    let size: Int64
    let parameterCount: String?
    let quantization: String?
    var isRemote: Bool = false
    
    var formattedSize: String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: size)
    }
}

struct ChatMessage: Codable, Sendable {
    let role: ChatRole
    let content: String
    
    enum ChatRole: String, Codable, Sendable {
        case system
        case user
        case assistant
    }
}

struct GenerationResult: Codable {
    let id: String
    let model: String
    let content: String
    let finishReason: FinishReason
    let promptTokens: Int
    let completionTokens: Int
    let totalTokens: Int
    
    enum FinishReason: String, Codable {
        case stop
        case length
    }
}

struct StreamChunk: Sendable {
    let token: String
    let tokenId: Int
    let logprob: Float?
    let finishReason: GenerationResult.FinishReason?
}

// MARK: - Errors

enum MLXError: LocalizedError, Sendable {
    case modelNotLoaded
    case loadFailed(String)
    case generationFailed(String)
    case modelNotFound
    case invalidRequest(String)
    
    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "Model is not loaded"
        case .loadFailed(let r): return "Load failed: \(r)"
        case .generationFailed(let r): return "Generation failed: \(r)"
        case .modelNotFound: return "Model not found"
        case .invalidRequest(let r): return "Invalid request: \(r)"
        }
    }
}
