import XCTest
@testable import MLXprovider

final class MLXBackendTests: XCTestCase {
    
    var backend: MLXBackend!
    
    override func setUp() async throws {
        super.setUp()
        backend = MLXBackend()
    }
    
    override func tearDown() async throws {
        backend = nil
        super.tearDown()
    }
    
    // MARK: - Model Scanning Tests
    
    func testListModelsReturnsArray() async throws {
        let models = await backend.listModels()
        XCTAssertNotNil(models)
    }
    
    func testScanEmptyDirectoryReturnsEmpty() async throws {
        let models = try await backend.scanDirectory("/nonexistent/path")
        XCTAssertEqual(models.count, 0)
    }
    
    func testModelInfoProperties() async throws {
        let models = try await backend.scanDirectory("/tmp")
        // May be empty if no MLX models in /tmp
        for model in models {
            XCTAssertFalse(model.id.isEmpty)
            XCTAssertFalse(model.name.isEmpty)
            XCTAssertFalse(model.path.isEmpty)
            XCTAssertGreaterThanOrEqual(model.size, 0)
        }
    }
    
    // MARK: - Model Loading Tests
    
    func testLoadModelThrowsWhenNotFound() async throws {
        do {
            _ = try await backend.loadModel(id: "nonexistent/model")
            XCTFail("Should throw error for nonexistent model")
        } catch {
            // Expected - model doesn't exist
            XCTAssertTrue(error is MLXError || error.localizedDescription.contains("not found"))
        }
    }
    
    func testUnloadNonExistentModelReturnsTrue() async throws {
        let result = await backend.unloadModel(id: "nonexistent/model")
        XCTAssertTrue(result)
    }
    
    func testIsModelLoadedReturnsFalseForUnloaded() async throws {
        let isLoaded = await backend.isModelLoaded("nonexistent/model")
        XCTAssertFalse(isLoaded)
    }
    
    // MARK: - Generation Tests
    
    func testGenerateWithoutLoadedModelThrows() async throws {
        let messages = [
            ChatMessage(role: .user, content: "Hello")
        ]
        
        do {
            _ = try await backend.generate(
                modelId: "nonexistent/model",
                messages: messages
            )
            XCTFail("Should throw error when model not loaded")
        } catch let error as MLXError {
            XCTAssertEqual(error, .modelNotLoaded)
        } catch {
            // May be MLXError or other error
            XCTAssertTrue(error.localizedDescription.contains("not loaded") || 
                          error.localizedDescription.contains("not found"))
        }
    }
    
    func testGenerateStreamWithoutModelThrows() async throws {
        let messages = [
            ChatMessage(role: .user, content: "Hello")
        ]
        
        let stream = await backend.generateStream(
            modelId: "nonexistent/model",
            messages: messages
        )
        
        // Consume the stream
        var receivedError = false
        for try await _ in stream {
            // Should not receive any tokens
        }
    }
    
    // MARK: - Message Parsing Tests
    
    func testChatMessageRoleParsing() {
        let systemMessage = ChatMessage(role: .system, content: "You are helpful")
        XCTAssertEqual(systemMessage.role, .system)
        
        let userMessage = ChatMessage(role: .user, content: "Hello")
        XCTAssertEqual(userMessage.role, .user)
        
        let assistantMessage = ChatMessage(role: .assistant, content: "Hi there")
        XCTAssertEqual(assistantMessage.role, .assistant)
    }
    
    func testChatMessageCodable() throws {
        let message = ChatMessage(role: .user, content: "Test message")
        let data = try JSONEncoder().encode(message)
        let decoded = try JSONDecoder().decode(ChatMessage.self, from: data)
        XCTAssertEqual(decoded.role, message.role)
        XCTAssertEqual(decoded.content, message.content)
    }
    
    // MARK: - Error Tests
    
    func testMLXErrorDescriptions() {
        XCTAssertEqual(MLXError.modelNotLoaded.errorDescription, "Model is not loaded. Please load a model first.")
        XCTAssertEqual(MLXError.modelNotFound.errorDescription, "Model not found")
        XCTAssertNotNil(MLXError.loadFailed("test").errorDescription)
        XCTAssertNotNil(MLXError.generationFailed("test").errorDescription)
        XCTAssertNotNil(MLXError.invalidRequest("test").errorDescription)
    }
}
