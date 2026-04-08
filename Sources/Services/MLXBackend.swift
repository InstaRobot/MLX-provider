import Foundation
import MLXLLM
import MLXLMCommon
import Tokenizers

// MARK: - MLX Backend Service

/// Pure Swift MLX backend for inference - no Python required
actor MLXBackend {
    private var loadedModels: [String: ModelContainer] = [:]
    private var loadedModelIds: [String] = []
    
    // HuggingFace token for model downloads
    private var hfToken: String?
    
    init() {}
    
    // MARK: - Configuration
    
    func setToken(_ token: String?) {
        self.hfToken = token
    }
    
    // MARK: - Model Discovery
    
    /// Get list of popular MLX community models
    func listModels() async -> [ModelInfo] {
        // Return a curated list of popular MLX community models
        let popularModels = [
            "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            "mlx-community/Qwen3-4B-4bit",
            "mlx-community/Qwen3-1.5B-4bit",
            "mlx-community/Gemma-2-2B-It-4bit",
            "mlx-community/Phi-4-mini-4bit",
            "mlx-community/Phi-3.5-mini-instruct-4bit",
            "mlx-community/OpenELM-3B-Instruct",
            "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
            "mlx-community/Mistral-Nemo-12B-4bit",
            "mlx-community/NousResearch-Hermes-3-8B-4bit"
        ]
        
        return popularModels.map { id in
            ModelInfo(
                id: id,
                name: id.replacingOccurrences(of: "mlx-community/", with: ""),
                path: "huggingface://\(id)",
                size: 0,
                parameterCount: extractParams(from: id),
                quantization: extractQuant(from: id),
                isRemote: true
            )
        }
    }
    
    private func extractParams(from modelId: String) -> String? {
        let patterns: [(String, Double)] = [
            ("70B", 70), ("65B", 65), ("40B", 40), ("34B", 34), ("30B", 30),
            ("22B", 22), ("13B", 13), ("12B", 12), ("11B", 11), ("8B", 8),
            ("7B", 7), ("4B", 4), ("3B", 3), ("2B", 2), ("1.5B", 1.5), ("1B", 1)
        ]
        for (pattern, _) in patterns {
            if modelId.contains(pattern) {
                return pattern
            }
        }
        return nil
    }
    
    private func extractQuant(from modelId: String) -> String? {
        if modelId.contains("4bit") { return "4-bit" }
        if modelId.contains("8bit") { return "8-bit" }
        return nil
    }
    
    // MARK: - Model Loading
    
    /// Load model from HuggingFace Hub (requires network)
    func loadModel(id: String) async throws -> Bool {
        if loadedModels[id] != nil {
            return true // Already loaded
        }
        
        do {
            print("[MLXBackend] Loading model: \(id)")
            
            // Create HuggingFace downloader
            let downloader = HFDownloader(token: hfToken)
            
            let container = try await loadModelContainer(
                from: downloader,
                using: TokenizersLoader(),
                id: id
            )
            
            loadedModels[id] = container
            loadedModelIds.append(id)
            
            print("[MLXBackend] Successfully loaded model: \(id)")
            return true
        } catch {
            print("[MLXBackend] Failed to load model \(id): \(error)")
            throw MLXError.loadFailed(error.localizedDescription)
        }
    }
    
    /// Load model from local directory
    func loadLocalModel(path: String) async throws -> Bool {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MLXError.modelNotFound
        }
        
        do {
            let container = try await loadModelContainer(
                from: url,
                using: TokenizersLoader()
            )
            
            let modelId = url.lastPathComponent
            loadedModels[modelId] = container
            loadedModelIds.append(modelId)
            
            print("[MLXBackend] Loaded local model: \(modelId)")
            return true
        } catch {
            throw MLXError.loadFailed(error.localizedDescription)
        }
    }
    
    func unloadModel(id: String) -> Bool {
        loadedModels.removeValue(forKey: id)
        loadedModelIds.removeAll { $0 == id }
        print("[MLXBackend] Unloaded model: \(id)")
        return true
    }
    
    func isModelLoaded(_ id: String) -> Bool {
        return loadedModels[id] != nil
    }
    
    func getLoadedModelId() -> String? {
        return loadedModelIds.first
    }
    
    // MARK: - Generation
    
    func generate(
        modelId: String,
        messages: [ChatMessage],
        temperature: Double = 0.7,
        maxTokens: Int = 512,
        repetitionPenalty: Double = 1.0
    ) async throws -> GenerationResult {
        guard let container = loadedModels[modelId] else {
            throw MLXError.modelNotLoaded
        }
        
        let prompt = buildPrompt(messages: messages)
        let inputIds = container.tokenizer.encode(prompt)
        let promptTokens = inputIds.count
        
        var outputIds: [Int] = []
        
        await container.model.perform { model in
            var token = inputIds
            for _ in 0..<maxTokens {
                let logits = model(token)
                let nextToken = sampleToken(logits: logits, temperature: Float(temperature))
                
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
        maxTokens: Int = 512,
        repetitionPenalty: Double = 1.0
    ) -> AsyncThrowingStream<StreamChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                guard let container = self.loadedModels[modelId] else {
                    continuation.finish(throwing: MLXError.modelNotLoaded)
                    return
                }
                
                let prompt = self.buildPrompt(messages: messages)
                let inputIds = container.tokenizer.encode(prompt)
                var outputIds: [Int] = []
                
                await container.model.perform { model in
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

// MARK: - HuggingFace Downloader

/// Simple HuggingFace Hub downloader for MLX models
struct HFDownloader: Downloader {
    let token: String?
    
    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        // Check cache first
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        let modelDir = cacheDir.appendingPathComponent("models/\(id)")
        
        if FileManager.default.fileExists(atPath: modelDir.path) {
            print("[HFDownloader] Using cached model: \(id)")
            return modelDir
        }
        
        // Download from HuggingFace Hub
        print("[HFDownloader] Downloading model: \(id)")
        
        // Construct API URL
        let revisionPart = revision != nil ? "/\(revision!)" : ""
        let apiUrl = "https://huggingface.co/api/models/\(id)\(revisionPart)"
        
        // Download model files
        let fileListUrl = "https://huggingface.co/\(id)/raw/main/.gitattributes"
        
        // For now, throw error - actual download implementation needed
        throw MLXError.loadFailed("HuggingFace download not implemented - use local models for now")
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
        case contentFilter = "content_filter"
        case modelSupport = "model_support"
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

public struct GenerateParameters: Sendable {
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
