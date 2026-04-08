import Foundation
import MLXLLM
import MLXLMHuggingFace
import MLXLMTokenizers

// MARK: - MLX Backend Service

/// Pure Swift MLX backend for inference - no Python required
actor MLXBackend {
    private var loadedModels: [String: LanguageModel] = [:]
    private var chatSessions: [String: ChatSession] = [:]
    private var modelDirectories: Set<String> = []
    
    private let modelCache: ModelCacheConfiguration
    
    init(cacheConfig: ModelCacheConfiguration = ModelCacheConfiguration()) {
        self.modelCache = cacheConfig
    }
    
    // MARK: - Model Scanning
    
    /// Scan a directory for MLX models
    func scanDirectory(_ path: String) throws -> [ModelInfo] {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return []
        }
        
        var models: [ModelInfo] = []
        let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        )
        
        while let fileURL = enumerator?.nextObject() as? URL {
            if isModelDirectory(fileURL) {
                if let modelInfo = extractModelInfo(from: fileURL) {
                    models.append(modelInfo)
                }
            }
        }
        
        // Also register for HuggingFace discovery
        modelDirectories.insert(path)
        
        return models
    }
    
    /// Get list of available models (local + HuggingFace registry)
    func listModels() -> [ModelInfo] {
        var allModels: [ModelInfo] = []
        
        // Local models
        for dir in modelDirectories {
            if let models = try? scanDirectory(dir) {
                allModels.append(contentsOf: models)
            }
        }
        
        // HuggingFace registry models
        let registryModels = LLMRegistry.all().map { config in
            ModelInfo(
                id: config.id,
                name: config.name ?? config.id,
                path: "huggingface://\(config.id)",
                size: 0,
                parameterCount: nil,
                quantization: config.id.contains("4bit") ? "4-bit" : nil,
                isRemote: true
            )
        }
        allModels.append(contentsOf: registryModels)
        
        return allModels
    }
    
    private func isModelDirectory(_ url: URL) -> Bool {
        let modelFiles = ["model.safetensors", "model.bin", "model.mlpack", "model.mlir"]
        let configFiles = ["config.json"]
        
        let hasModelFile = modelFiles.contains { file in
            FileManager.default.fileExists(atPath: url.appendingPathComponent(file).path)
        }
        let hasConfigFile = configFiles.contains { file in
            FileManager.default.fileExists(atPath: url.appendingPathComponent(file).path)
        }
        
        return hasModelFile || hasConfigFile
    }
    
    private func extractModelInfo(from url: URL) -> ModelInfo? {
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
            
            // Estimate parameter count
            if let hiddenSize = config["hidden_size"] as? Int,
               let numLayers = config["num_hidden_layers"] as? Int,
               let numHeads = config["num_attention_heads"] as? Int {
                // Rough estimation: 12 * hidden_size^2 * num_layers
                let approx = Double(hiddenSize * hiddenSize * numLayers * 12) / 1e9
                if approx > 1 {
                    parameterCount = String(format: "%.1fB", approx)
                } else {
                    parameterCount = String(format: "%.0fM", approx * 1000)
                }
            }
            
            // Quantization info
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
    
    // MARK: - Model Management
    
    func loadModel(id: String) async throws -> Bool {
        if loadedModels[id] != nil {
            return true // Already loaded
        }
        
        do {
            let tokenizerLoader = TokenizersLoader()
            let model = try await loadModel(
                using: tokenizerLoader,
                id: id,
                cache: modelCache
            )
            
            loadedModels[id] = model
            chatSessions[id] = ChatSession(model)
            
            print("[MLXBackend] Loaded model: \(id)")
            return true
        } catch {
            print("[MLXBackend] Failed to load model \(id): \(error)")
            throw MLXError.loadFailed(error.localizedDescription)
        }
    }
    
    func unloadModel(id: String) -> Bool {
        loadedModels.removeValue(forKey: id)
        chatSessions.removeValue(forKey: id)
        print("[MLXBackend] Unloaded model: \(id)")
        return true
    }
    
    func isModelLoaded(_ id: String) -> Bool {
        return loadedModels[id] != nil
    }
    
    func getLoadedModelId() -> String? {
        return loadedModels.keys.first
    }
    
    // MARK: - Generation
    
    func generate(
        modelId: String,
        messages: [ChatMessage],
        temperature: Double = 0.7,
        maxTokens: Int = 512,
        repetitionPenalty: Double = 1.0,
        stopWords: [String]? = nil
    ) async throws -> GenerationResult {
        guard let model = loadedModels[modelId] else {
            throw MLXError.modelNotLoaded
        }
        
        let session = chatSessions[modelId] ?? ChatSession(model)
        
        // Convert messages to prompt
        let prompt = buildPrompt(messages: messages)
        
        // Generate
        let params = GenerateParameters(
            temperature: Float(temperature),
            maxTokens: maxTokens,
            repetitionPenalty: Float(repetitionPenalty)
        )
        
        let stream = try await model.generate(
            prompt: prompt,
            parameters: params
        )
        
        var fullResponse = ""
        var tokenCount = 0
        
        for try await token in stream {
            fullResponse += token.tokenText
            tokenCount += 1
        }
        
        return GenerationResult(
            id: "chatcmpl-\(UUID().uuidString.prefix(8))",
            model: modelId,
            content: fullResponse.trimmingCharacters(in: .whitespacesAndNewlines),
            finishReason: .stop,
            promptTokens: countTokens(prompt),
            completionTokens: tokenCount,
            totalTokens: countTokens(prompt) + tokenCount
        )
    }
    
    /// Streaming generation - returns AsyncStream for real-time token output
    func generateStream(
        modelId: String,
        messages: [ChatMessage],
        temperature: Double = 0.7,
        maxTokens: Int = 512,
        repetitionPenalty: Double = 1.0,
        stopWords: [String]? = nil
    ) -> AsyncThrowingStream<StreamChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                guard let model = self.loadedModels[modelId] else {
                    continuation.finish(throwing: MLXError.modelNotLoaded)
                    return
                }
                
                let prompt = self.buildPrompt(messages: messages)
                let params = GenerateParameters(
                    temperature: Float(temperature),
                    maxTokens: maxTokens,
                    repetitionPenalty: Float(repetitionPenalty)
                )
                
                do {
                    let stream = try await model.generate(prompt: prompt, parameters: params)
                    
                    for try await token in stream {
                        let chunk = StreamChunk(
                            token: token.tokenText,
                            tokenId: Int(token.tokenId),
                            logprob: token.logProbability,
                            finishReason: token.finishReason.map { .init(rawValue: $0.rawValue) }
                        )
                        continuation.yield(chunk)
                    }
                    
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    // MARK: - Helper Methods
    
    private func buildPrompt(messages: [ChatMessage]) -> String {
        // Simple prompt building - can be enhanced with chat templates
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
    
    private func countTokens(_ text: String) -> Int {
        // Rough estimation: ~4 characters per token for English
        return max(1, text.count / 4)
    }
}

// MARK: - Data Types

struct ModelInfo: Identifiable, Codable, Hashable {
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

struct ChatMessage: Codable {
    let role: ChatRole
    let content: String
    
    enum ChatRole: String, Codable {
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
        case contentFilter = "content_filter"
        case modelSupport = "model_support"
    }
}

struct StreamChunk {
    let token: String
    let tokenId: Int
    let logprob: Float?
    let finishReason: GenerationResult.FinishReason?
}

// MARK: - Model Cache Configuration

public struct ModelCacheConfiguration {
    public let memoryLimit: Int?
    public let maxKVCache: Int?
    
    public init(memoryLimit: Int? = nil, maxKVCache: Int? = nil) {
        self.memoryLimit = memoryLimit
        self.maxKVCache = maxKVCache
    }
}

// MARK: - Errors

enum MLXError: LocalizedError {
    case modelNotLoaded
    case loadFailed(String)
    case generationFailed(String)
    case modelNotFound
    case invalidRequest(String)
    
    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Model is not loaded. Please load a model first."
        case .loadFailed(let reason):
            return "Failed to load model: \(reason)"
        case .generationFailed(let reason):
            return "Generation failed: \(reason)"
        case .modelNotFound:
            return "Model not found"
        case .invalidRequest(let reason):
            return "Invalid request: \(reason)"
        }
    }
}

// MARK: - Generate Parameters

public struct GenerateParameters {
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var minP: Float
    public var maxTokens: Int
    public var repetitionPenalty: Float
    public var repetitionContextSize: Int
    
    public init(
        temperature: Float = 0.7,
        topP: Float = 0.9,
        topK: Int = 50,
        minP: Float = 0.0,
        maxTokens: Int = 512,
        repetitionPenalty: Float = 1.0,
        repetitionContextSize: Int = 20
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.maxTokens = maxTokens
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
    }
}

// MARK: - OpenAI API Compatibility

extension MLXBackend {
    
    /// Convert OpenAI-style chat completion request to internal format
    func handleChatCompletion(
        body: Data
    ) async throws -> Data {
        guard let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let modelId = json["model"] as? String,
              let messagesData = json["messages"] as? [[String: Any]] else {
            throw MLXError.invalidRequest("Missing required fields: model, messages")
        }
        
        let messages = messagesData.compactMap { dict -> ChatMessage? in
            guard let roleStr = dict["role"] as? String,
                  let content = dict["content"] as? String,
                  let role = ChatMessage.ChatRole(rawValue: roleStr) else {
                return nil
            }
            return ChatMessage(role: role, content: content)
        }
        
        let temperature = json["temperature"] as? Double ?? 0.7
        let maxTokens = json["max_tokens"] as? Int ?? 512
        let stream = json["stream"] as? Bool ?? false
        
        if stream {
            // Return streaming response
            let streamChunks = generateStream(
                modelId: modelId,
                messages: messages,
                temperature: temperature,
                maxTokens: maxTokens
            )
            
            return try await encodeStreamResponse(
                streamChunks: streamChunks,
                modelId: modelId
            )
        } else {
            // Return regular response
            let result = try await generate(
                modelId: modelId,
                messages: messages,
                temperature: temperature,
                maxTokens: maxTokens
            )
            
            return try encodeResponse(result: result)
        }
    }
    
    private func encodeResponse(result: GenerationResult) throws -> Data {
        let response: [String: Any] = [
            "id": result.id,
            "object": "chat.completion",
            "created": Int(Date().timeIntervalSince1970),
            "model": result.model,
            "choices": [
                [
                    "index": 0,
                    "message": [
                        "role": "assistant",
                        "content": result.content
                    ],
                    "finish_reason": result.finishReason.rawValue
                ]
            ],
            "usage": [
                "prompt_tokens": result.promptTokens,
                "completion_tokens": result.completionTokens,
                "total_tokens": result.totalTokens
            ]
        ]
        
        return try JSONSerialization.data(withJSONObject: response)
    }
    
    private func encodeStreamResponse(
        streamChunks: AsyncThrowingStream<StreamChunk, Error>,
        modelId: String
    ) async throws -> Data {
        // For streaming, we return SSE format
        // This is simplified - real implementation would use EventStream
        var chunks: [[String: Any]] = []
        let id = "chatcmpl-\(UUID().uuidString.prefix(8))"
        let created = Int(Date().timeIntervalSince1970)
        
        for try await chunk in streamChunks {
            let chunkData: [String: Any] = [
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": modelId,
                "choices": [
                    [
                        "index": 0,
                        "delta": ["content": chunk.token],
                        "finish_reason": chunk.finishReason?.rawValue as Any
                    ]
                ]
            ]
            chunks.append(chunkData)
        }
        
        return try JSONSerialization.data(withJSONObject: chunks)
    }
}
