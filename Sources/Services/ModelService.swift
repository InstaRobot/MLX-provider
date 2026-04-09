import Foundation
import MLXLLM
import MLXLMCommon
import Tokenizers

// MARK: - Chat Message Types

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

// MARK: - Local Tokenizer Loader

struct LocalTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        return TokenizerBridge(inner: tokenizer)
    }
}

// MARK: - Tokenizer Bridge

struct TokenizerBridge: MLXLMCommon.Tokenizer {
    let inner: Tokenizers.Tokenizer

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        inner.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        inner.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        inner.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        inner.convertIdToToken(id)
    }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        // Simple concatenation for now
        var result: [Int] = []
        for msg in messages {
            if let role = msg["role"] as? String, let content = msg["content"] as? String {
                result.append(contentsOf: inner.encode(text: "\(role): \(content)", addSpecialTokens: false))
            }
        }
        return result
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
    let contextWindow: Int?

    init(id: String, name: String, path: String, size: Int64, parameterCount: String? = nil, quantization: String? = nil, contextWindow: Int? = nil) {
        self.id = id
        self.name = name
        self.path = path
        self.size = size
        self.parameterCount = parameterCount
        self.quantization = quantization
        self.contextWindow = contextWindow
    }
}

// MARK: - Model Service Actor

@MainActor
final class MLXModelManager: ObservableObject {
    @Published var isModelLoaded = false
    @Published var isLoading = false
    @Published var errorMessage: String?

    private var container: ModelContainer?

    // MARK: - Model Loading

    func loadModel(modelInfo: ModelInfo) async throws {
        isLoading = true
        errorMessage = nil

        do {
            let config = ModelConfiguration(
                directory: URL(fileURLWithPath: modelInfo.path)
            )

            let loader = LocalTokenizerLoader()
            let container = try await LLMModelFactory.shared.loadContainer(
                from: URL(fileURLWithPath: modelInfo.path),
                using: loader
            )

            self.container = container

            isModelLoaded = true
            isLoading = false
        } catch {
            isLoading = false
            errorMessage = error.localizedDescription
            throw error
        }
    }

    func unloadModel() {
        container = nil
        isModelLoaded = false
    }

    // MARK: - Generation

    func generate(messages: [ChatMessage]) async throws -> GenerationResult {
        guard let container = container else {
            throw MLXModelError.modelNotLoaded
        }

        let userText = messages.last?.content ?? ""
        let userInput = UserInput(prompt: userText)

        let lmInput = try await container.prepare(input: userInput)
        let params = GenerateParameters(maxTokens: 2048, temperature: 0.7)

        var fullResponse = ""
        for await event in try await container.generate(input: lmInput, parameters: params) {
            if case .chunk(let text) = event {
                fullResponse += text
            }
        }

        return GenerationResult(
            id: "chatcmpl-\(UUID().uuidString.prefix(8))",
            model: "local-mlx",
            content: fullResponse,
            finishReason: .stop,
            promptTokens: 0,
            completionTokens: 0,
            totalTokens: 0
        )
    }

    func generateStream(messages: [ChatMessage]) -> AsyncThrowingStream<StreamChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                guard let container = self.container else {
                    continuation.finish(throwing: MLXModelError.modelNotLoaded)
                    return
                }

                let userText = messages.last?.content ?? ""
                let userInput = UserInput(prompt: userText)

                do {
                    let lmInput = try await container.prepare(input: userInput)
                    let params = GenerateParameters(maxTokens: 2048, temperature: 0.7)

                    for await event in try await container.generate(input: lmInput, parameters: params) {
                        if case .chunk(let text) = event {
                            let streamChunk = StreamChunk(
                                token: text,
                                tokenId: 0,
                                logprob: nil,
                                finishReason: nil
                            )
                            continuation.yield(streamChunk)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}

// MARK: - Errors

enum MLXModelError: LocalizedError, Sendable {
    case modelNotLoaded
    case loadFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "Model is not loaded"
        case .loadFailed(let r): return "Load failed: \(r)"
        }
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
        let fileManager = FileManager.default

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
        var contextWindow: Int? = nil

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
            if let maxPosEmb = config["max_position_embeddings"] as? Int {
                contextWindow = maxPosEmb
            } else if let modelMaxLen = config["model_max_length"] as? Int {
                contextWindow = modelMaxLen
            }
        }

        return ModelInfo(
            id: name,
            name: name,
            path: path,
            size: totalSize,
            parameterCount: parameterCount,
            quantization: quantization,
            contextWindow: contextWindow
        )
    }
}
