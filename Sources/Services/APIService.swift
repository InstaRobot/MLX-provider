import Foundation
import Network

actor APIService {
    private var listener: NWListener?
    private var modelService: ModelService?
    private var port: Int = 8080
    private var isRunning = false
    
    enum HTTPMethod: String {
        case GET, POST, PUT, DELETE, OPTIONS
    }
    
    struct HTTPRequest {
        let method: HTTPMethod
        let path: String
        let headers: [String: String]
        let body: Data?
    }
    
    struct HTTPResponse {
        let statusCode: Int
        let headers: [String: String]
        let body: Data?
        
        init(statusCode: Int, headers: [String: String] = [:], body: Data?) {
            self.statusCode = statusCode
            self.headers = headers.merging(["server": "MLX-provider/1.0"]) { $1 }
            self.body = body
        }
    }
    
    // MARK: - Server Lifecycle
    
    func start(port: Int, modelService: ModelService) async throws {
        self.port = port
        self.modelService = modelService
        
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse
        
        let nwPort = NWEndpoint.Port(integerLiteral: UInt16(port))
        listener = try NWListener(using: parameters, on: nwPort)
        
        listener?.stateUpdateHandler = { [weak self] state in
            Task { await self?.handleListenerState(state) }
        }
        
        listener?.newConnectionHandler = { [weak self] connection in
            Task { await self?.handleConnection(connection) }
        }
        
        listener?.start(queue: .global())
        isRunning = true
    }
    
    func stop() async {
        listener?.cancel()
        listener = nil
        isRunning = false
    }
    
    private func handleListenerState(_ state: NWListener.State) {
        switch state {
        case .ready:
            print("MLX-provider server listening on port \(port)")
        case .failed(let error):
            print("Server failed: \(error)")
        case .cancelled:
            print("Server cancelled")
        default:
            break
        }
    }
    
    // MARK: - Connection Handling
    
    private func handleConnection(_ connection: NWConnection) async {
        connection.stateUpdateHandler = { state in
            switch state {
            case .ready:
                Task { await self.receiveRequest(connection) }
            case .failed(let error):
                print("Connection failed: \(error)")
            default:
                break
            }
        }
        connection.start(queue: .global())
    }
    
    private func receiveRequest(_ connection: NWConnection) async {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            guard let self = self, let data = data, !data.isEmpty else {
                connection.cancel()
                return
            }
            
            Task {
                if let response = await self.processRequest(data) {
                    await self.sendResponse(connection, response: response)
                } else {
                    await self.sendResponse(connection, response: HTTPResponse(statusCode: 500, headers: [:], body: nil))
                }
            }
        }
    }
    
    private func sendResponse(_ connection: NWConnection, response: HTTPResponse) async {
        let statusLine = "HTTP/1.1 \(response.statusCode) \(self.httpStatusText(response.statusCode))\r\n"
        var headerStr = statusLine
        for (key, value) in response.headers {
            headerStr += "\(key): \(value)\r\n"
        }
        headerStr += "Content-Length: \(response.body?.count ?? 0)\r\n"
        headerStr += "Access-Control-Allow-Origin: *\r\n"
        headerStr += "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        headerStr += "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        headerStr += "\r\n"
        
        var responseData = Data(headerStr.utf8)
        if let body = response.body {
            responseData.append(body)
        }
        
        connection.send(content: responseData, completion: .contentProcessed { _ in
            connection.cancel()
        })
    }
    
    // MARK: - Request Processing
    
    private func processRequest(_ data: Data) async -> HTTPResponse? {
        guard let requestString = String(data: data, encoding: .utf8) else {
            return nil
        }
        
        let lines = requestString.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else {
            return nil
        }
        
        let parts = requestLine.components(separatedBy: " ")
        guard parts.count >= 2 else {
            return nil
        }
        
        let methodStr = parts[0]
        let path = parts[1]
        
        guard let method = HTTPMethod(rawValue: methodStr) else {
            return HTTPResponse(statusCode: 400, headers: [:], body: nil)
        }
        
        // Parse headers
        var headers: [String: String] = [:]
        var bodyStartIndex = 0
        for (index, line) in lines.enumerated() {
            if line.isEmpty {
                bodyStartIndex = index + 1
                break
            }
            if index > 0 && index < lines.count - 1 {
                let headerParts = line.components(separatedBy: ": ")
                if headerParts.count == 2 {
                    headers[headerParts[0]] = headerParts[1]
                }
            }
        }
        
        // Extract body
        let bodyData: Data? = bodyStartIndex < lines.count
            ? data(using: .utf8)?.subdata(in: data.rangeOf("\r\n\r\n", options: .literal)?.upperBound ?? data.startIndex..<data.endIndex)
            : nil
        
        return await routeRequest(method: method, path: path, headers: headers, body: bodyData)
    }
    
    private func routeRequest(method: HTTPMethod, path: String, headers: [String: String], body: Data?) async -> HTTPResponse? {
        // Route to handlers
        switch path {
        case "/v1/models":
            if method == .GET {
                return await handleListModels()
            }
        case "/v1/chat/completions":
            if method == .POST {
                return await handleChatCompletions(body: body)
            }
        case "/v1/models/load":
            if method == .POST {
                return await handleLoadModel(body: body)
            }
        default:
            if path.hasPrefix("/v1/models/") && path.hasSuffix("/load") && method == .POST {
                let modelId = path.replacingOccurrences(of: "/v1/models/", with: "").replacingOccurrences(of: "/load", with: "")
                return await handleSpecificModelLoad(modelId: modelId, body: body)
            }
        }
        
        return HTTPResponse(statusCode: 404, headers: [:], body: "{ \"error\": \"Not Found\" }".data(using: .utf8))
    }
    
    // MARK: - API Handlers
    
    private func handleListModels() async -> HTTPResponse {
        guard let modelService = modelService else {
            return HTTPResponse(statusCode: 500, headers: [:], body: "{ \"error\": \"No model service\" }".data(using: .utf8))
        }
        
        do {
            let models = try await modelService.scanDirectory("~/Models/mlx")
            
            let response: [String: Any] = [
                "object": "list",
                "data": models.map { model in
                    [
                        "id": model.id,
                        "object": "model",
                        "created": Int(Date().timeIntervalSince1970),
                        "owned_by": "local",
                        "root": model.path,
                        "parent": NSNull()
                    ]
                }
            ]
            
            let jsonData = try JSONSerialization.data(withJSONObject: response)
            return HTTPResponse(statusCode: 200, headers: ["Content-Type": "application/json"], body: jsonData)
        } catch {
            return HTTPResponse(statusCode: 500, headers: [:], body: "{ \"error\": \"\(error.localizedDescription)\" }".data(using: .utf8))
        }
    }
    
    private func handleChatCompletions(body: Data?) async -> HTTPResponse {
        guard let body = body else {
            return HTTPResponse(statusCode: 400, headers: [:], body: "{ \"error\": \"Missing body\" }".data(using: .utf8))
        }
        
        guard let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let modelId = json["model"] as? String,
              let messages = json["messages"] as? [[String: Any]] else {
            return HTTPResponse(statusCode: 400, headers: [:], body: "{ \"error\": \"Invalid request\" }".data(using: .utf8))
        }
        
        let temperature = json["temperature"] as? Double ?? 0.7
        let maxTokens = json["max_tokens"] as? Int ?? 512
        let stream = json["stream"] as? Bool ?? false
        
        // Forward to mlx_lm server
        let mlxResponse = await forwardToMLXServer(
            modelId: modelId,
            messages: messages,
            temperature: temperature,
            maxTokens: maxTokens,
            stream: stream
        )
        
        return HTTPResponse(statusCode: 200, headers: ["Content-Type": "application/json"], body: mlxResponse)
    }
    
    private func handleLoadModel(body: Data?) async -> HTTPResponse {
        guard let modelService = modelService else {
            return HTTPResponse(statusCode: 500, headers: [:], body: "{ \"error\": \"No model service\" }".data(using: .utf8))
        }
        
        guard let body = body,
              let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let modelId = json["model"] as? String else {
            return HTTPResponse(statusCode: 400, headers: [:], body: "{ \"error\": \"Invalid request\" }".data(using: .utf8))
        }
        
        do {
            try await modelService.loadModel(modelId)
            return HTTPResponse(statusCode: 200, headers: [:], body: "{ \"status\": \"loaded\", \"model\": \"\(modelId)\" }".data(using: .utf8))
        } catch {
            return HTTPResponse(statusCode: 500, headers: [:], body: "{ \"error\": \"\(error.localizedDescription)\" }".data(using: .utf8))
        }
    }
    
    private func handleSpecificModelLoad(modelId: String, body: Data?) async -> HTTPResponse {
        return await handleLoadModel(body: body)
    }
    
    private func forwardToMLXServer(modelId: String, messages: [[String: Any]], temperature: Double, maxTokens: Int, stream: Bool) async -> Data? {
        // This would communicate with the mlx_lm Python server
        // For now, return a placeholder - actual implementation would use HTTP
        let requestBody: [String: Any] = [
            "model": modelId,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": maxTokens,
            "stream": stream
        ]
        
        guard let jsonData = try? JSONSerialization.data(withJSONObject: requestBody) else {
            return nil
        }
        
        // TODO: Forward to mlx_lm server at localhost:port
        // This is a placeholder for actual mlx_lm integration
        
        let response: [String: Any] = [
            "id": "chatcmpl-\(UUID().uuidString.prefix(8))",
            "object": "chat.completion",
            "created": Int(Date().timeIntervalSince1970),
            "model": modelId,
            "choices": [
                [
                    "index": 0,
                    "message": [
                        "role": "assistant",
                        "content": "MLX-provider is running. Configure your models directory and start the server."
                    ],
                    "finish_reason": "stop"
                ]
            ],
            "usage": [
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            ]
        ]
        
        return try? JSONSerialization.data(withJSONObject: response)
    }
    
    // MARK: - Utilities
    
    private func httpStatusText(_ code: Int) -> String {
        switch code {
        case 200: return "OK"
        case 400: return "Bad Request"
        case 401: return "Unauthorized"
        case 403: return "Forbidden"
        case 404: return "Not Found"
        case 500: return "Internal Server Error"
        default: return "Unknown"
        }
    }
}
