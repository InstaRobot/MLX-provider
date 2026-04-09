import Foundation
import MLX
import MLXLLM
import MLXLMCommon

// MARK: - MLX Backend Service

/// MLX Backend that communicates with the Python mlx_lm server
/// for model inference on Apple Silicon.
actor MLXBackend {
    private var loadedModelId: String?
    private let serverURL: String

    init(port: Int = 8080) {
        self.serverURL = "http://localhost:\(port)"
    }

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
        let configURL = url.appendingPathComponent("config.json")
        let hasConfig = FileManager.default.fileExists(atPath: configURL.path)

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

        var totalSize: Int64 = 0
        if let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) {
            while let fileURL = enumerator.nextObject() as? URL {
                if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                    totalSize += Int64(size)
                }
            }
        }

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
            quantization: quantization
        )
    }

    /// List available local models
    func listModels() async -> [ModelInfo] {
        return []
    }

    // MARK: - Model Loading

    /// Load model via HTTP API
    func loadModel(path: String) async throws -> Bool {
        let modelId = URL(fileURLWithPath: path).lastPathComponent

        guard let url = URL(string: "\(serverURL)/v1/models/\(modelId)/load") else {
            throw MLXError.invalidRequest("Invalid server URL")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONEncoder().encode(["model": modelId])

        let (_, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw MLXError.loadFailed("Failed to load model")
        }

        loadedModelId = modelId
        return true
    }

    /// Load model by ID
    func loadModel(id: String) async throws -> Bool {
        return try await loadModel(path: id)
    }

    func unloadModel(id: String) -> Bool {
        if loadedModelId == id || loadedModelId == nil {
            loadedModelId = nil
            return true
        }
        return false
    }

    func isModelLoaded(_ id: String) -> Bool {
        return loadedModelId == id
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
        guard let url = URL(string: "\(serverURL)/v1/chat/completions") else {
            throw MLXError.invalidRequest("Invalid server URL")
        }

        let chatMessages = messages.map { ["role": $0.role.rawValue, "content": $0.content] }

        let requestBody: [String: Any] = [
            "model": modelId,
            "messages": chatMessages,
            "temperature": temperature,
            "max_tokens": maxTokens,
            "stream": false
        ]

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw MLXError.generationFailed("Generation request failed")
        }

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = json["choices"] as? [[String: Any]],
              let firstChoice = choices.first,
              let message = firstChoice["message"] as? [String: Any],
              let content = message["content"] as? String else {
            throw MLXError.generationFailed("Invalid response format")
        }

        let id = json["id"] as? String ?? "chatcmpl-\(UUID().uuidString.prefix(8))"
        let usage = json["usage"] as? [String: Any] ?? [:]
        let promptTokens = usage["prompt_tokens"] as? Int ?? 0
        let completionTokens = usage["completion_tokens"] as? Int ?? 0

        return GenerationResult(
            id: id,
            model: modelId,
            content: content.trimmingCharacters(in: .whitespacesAndNewlines),
            finishReason: .stop,
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            totalTokens: promptTokens + completionTokens
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
                do {
                    guard let url = URL(string: "\(self.serverURL)/v1/chat/completions") else {
                        throw MLXError.invalidRequest("Invalid server URL")
                    }

                    let chatMessages = messages.map { ["role": $0.role.rawValue, "content": $0.content] }

                    let requestBody: [String: Any] = [
                        "model": modelId,
                        "messages": chatMessages,
                        "temperature": temperature,
                        "max_tokens": maxTokens,
                        "stream": true
                    ]

                    var request = URLRequest(url: url)
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)

                    let (data, response) = try await URLSession.shared.data(for: request)

                    guard let httpResponse = response as? HTTPURLResponse,
                          (200...299).contains(httpResponse.statusCode) else {
                        throw MLXError.generationFailed("Stream request failed")
                    }

                    guard let responseText = String(data: data, encoding: .utf8) else {
                        throw MLXError.generationFailed("Invalid response encoding")
                    }

                    // Parse SSE stream
                    let lines = responseText.components(separatedBy: "\n")
                    for line in lines {
                        if line.hasPrefix("data: ") {
                            let jsonStr = String(line.dropFirst(6))
                            if jsonStr == "[DONE]" {
                                continuation.finish()
                                return
                            }
                            if let jsonData = jsonStr.data(using: .utf8),
                               let json = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
                               let choices = json["choices"] as? [[String: Any]],
                               let delta = choices.first?["delta"] as? [String: Any],
                               let content = delta["content"] as? String {
                                let chunk = StreamChunk(
                                    token: content,
                                    tokenId: 0,
                                    logprob: nil,
                                    finishReason: nil
                                )
                                continuation.yield(chunk)
                            }
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

// MARK: - Data Types

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