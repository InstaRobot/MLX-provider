import XCTest
@testable import MLXprovider

final class APIServiceTests: XCTestCase {
    
    // MARK: - HTTP Request Parsing Tests
    
    func testParseSimpleGETRequest() {
        let requestData = """
        GET /v1/models HTTP/1.1
        Host: localhost:8080
        
        """.data(using: .utf8)!
        
        let parsed = HTTPParser.parse(requestData)
        XCTAssertNotNil(parsed)
        XCTAssertEqual(parsed?.method, .GET)
        XCTAssertEqual(parsed?.path, "/v1/models")
    }
    
    func testParsePOSTRequestWithBody() {
        let requestData = """
        POST /v1/chat/completions HTTP/1.1
        Host: localhost:8080
        Content-Type: application/json
        Content-Length: 78
        
        {"model":"test-model","messages":[{"role":"user","content":"Hello"}]}
        """.data(using: .utf8)!
        
        let parsed = HTTPParser.parse(requestData)
        XCTAssertNotNil(parsed)
        XCTAssertEqual(parsed?.method, .POST)
        XCTAssertEqual(parsed?.path, "/v1/chat/completions")
        XCTAssertEqual(parsed?.headers["Content-Type"], "application/json")
    }
    
    func testParseRequestWithNoBody() {
        let requestData = """
        DELETE /v1/models/test-model HTTP/1.1
        Host: localhost:8080
        
        """.data(using: .utf8)!
        
        let parsed = HTTPParser.parse(requestData)
        XCTAssertNotNil(parsed)
        XCTAssertEqual(parsed?.method, .DELETE)
        XCTAssertEqual(parsed?.path, "/v1/models/test-model")
    }
    
    // MARK: - HTTP Response Tests
    
    func testResponseWithStatusCode() {
        let response = HTTPResponse(statusCode: 200, body: "OK".data(using: .utf8))
        XCTAssertEqual(response.statusCode, 200)
        XCTAssertNotNil(response.body)
    }
    
    func testResponseWithJSONBody() {
        let json: [String: Any] = ["key": "value"]
        let body = try? JSONSerialization.data(withJSONObject: json)
        let response = HTTPResponse(statusCode: 200, headers: ["Content-Type": "application/json"], body: body)
        
        XCTAssertEqual(response.statusCode, 200)
        XCTAssertEqual(response.headers["Content-Type"], "application/json")
    }
    
    func testResponseStatusText() {
        XCTAssertEqual(HTTPResponse.statusText(for: 200), "OK")
        XCTAssertEqual(HTTPResponse.statusText(for: 400), "Bad Request")
        XCTAssertEqual(HTTPResponse.statusText(for: 404), "Not Found")
        XCTAssertEqual(HTTPResponse.statusText(for: 500), "Internal Server Error")
    }
    
    // MARK: - Routing Tests
    
    func testRouteToListModels() {
        let route = HTTPRouter.route("/v1/models", method: .GET)
        XCTAssertEqual(route, .listModels)
    }
    
    func testRouteToChatCompletions() {
        let route = HTTPRouter.route("/v1/chat/completions", method: .POST)
        XCTAssertEqual(route, .chatCompletions)
    }
    
    func testRouteToModelLoad() {
        let route = HTTPRouter.route("/v1/models/test-model/load", method: .POST)
        XCTAssertEqual(route, .loadModel("test-model"))
    }
    
    func testRouteToModelUnload() {
        let route = HTTPRouter.route("/v1/models/test-model/unload", method: .POST)
        XCTAssertEqual(route, .unloadModel("test-model"))
    }
    
    func testRouteUnknownReturnsNotFound() {
        let route = HTTPRouter.route("/unknown/path", method: .GET)
        XCTAssertEqual(route, .notFound)
    }
    
    // MARK: - API Request/Response Tests
    
    func testChatCompletionRequestParsing() throws {
        let requestJSON = """
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
        """.data(using: .utf8)!
        
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: requestJSON)
        XCTAssertEqual(request.model, "test-model")
        XCTAssertEqual(request.messages.count, 2)
        XCTAssertEqual(request.temperature, 0.7)
        XCTAssertEqual(request.maxTokens, 100)
    }
    
    func testChatCompletionResponseEncoding() throws {
        let response = ChatCompletionResponse(
            id: "test-123",
            object: "chat.completion",
            created: 1234567890,
            model: "test-model",
            choices: [
                Choice(
                    index: 0,
                    message: Message(role: .assistant, content: "Hello!"),
                    finishReason: .stop
                )
            ],
            usage: Usage(
                promptTokens: 10,
                completionTokens: 5,
                totalTokens: 15
            )
        )
        
        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode([String: Any].self, from: data)
        
        XCTAssertEqual(decoded["id"] as? String, "test-123")
        XCTAssertEqual(decoded["model"] as? String, "test-model")
    }
    
    // MARK: - CORS Tests
    
    func testCORSHeadersPresent() {
        let response = HTTPResponse(statusCode: 200, body: nil)
        XCTAssertEqual(response.headers["Access-Control-Allow-Origin"], "*")
        XCTAssertEqual(response.headers["Access-Control-Allow-Methods"], "GET, POST, OPTIONS")
    }
}

// MARK: - Test Helpers

extension JSONDecoder {
    convenience init() {
        self.init()
    }
}
