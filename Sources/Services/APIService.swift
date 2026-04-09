import Foundation
import Network

actor APIService {
    private var listener: NWListener?
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

    func start(port: Int) async throws {
        self.port = port

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
        var bodyData: Data? = nil
        if bodyStartIndex < lines.count {
            let bodyLines = lines[bodyStartIndex...]
            let bodyString = bodyLines.joined(separator: "\r\n")
            bodyData = bodyString.data(using: .utf8)
        }

        return await routeRequest(method: method, path: path, headers: headers, body: bodyData)
    }

    private func routeRequest(method: HTTPMethod, path: String, headers: [String: String], body: Data?) async -> HTTPResponse? {
        switch path {
        case "/v1/models":
            if method == .GET {
                return HTTPResponse(statusCode: 200, headers: ["Content-Type": "application/json"], body: "{ \"object\": \"list\", \"data\": [] }".data(using: .utf8))
            }
        case "/v1/chat/completions":
            if method == .POST {
                return HTTPResponse(statusCode: 200, headers: ["Content-Type": "application/json"], body: "{ \"error\": \"Chat is handled directly via MLX\" }".data(using: .utf8))
            }
        default:
            break
        }

        return HTTPResponse(statusCode: 404, headers: [:], body: "{ \"error\": \"Not Found\" }".data(using: .utf8))
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
