# MLX-provider Specification

## Overview

MLX-provider is a desktop application that provides an **OpenAI-compatible REST API** for MLX models on Apple Silicon. It acts as a local inference server, similar to Ollama, but purpose-built for Apple's MLX framework.

## Core Features

### 1. Model Management
- **Model Scanning**: Automatically discover MLX models in a configurable directory
- **Model Loading**: Load/unload models into GPU memory on demand
- **Model Selection**: Switch between available models via API or UI
- **Model Info**: Expose model metadata (parameters, quantization, context length)

### 2. OpenAI-Compatible API
Implements the OpenAI Chat Completions API format:

```
POST /v1/chat/completions
GET  /v1/models
POST /v1/completions (optional)
```

**Endpoints:**
- `GET /v1/models` — list available models
- `POST /v1/chat/completions` — chat completion
- `POST /v1/models/{model}/load` — preload a model
- `POST /v1/models/{model}/unload` — unload a model

### 3. Model Directory Configuration
- User specifies a folder path where MLX models are stored
- Supported formats: `.mlx`, directories with `model.safetensors` + config
- Recursive scanning with caching

### 4. UI (SwiftUI Desktop App)
- Model directory picker
- Current model status (loaded/unloaded)
- API server status (running/stopped)
- Server port configuration
- API key authentication (optional)
- Simple logs viewer

## Technical Stack

- **Language**: Swift 6.0+
- **UI Framework**: SwiftUI
- **Platform**: macOS 15+
- **MLX Integration**: Via Swift-MLX (if available) or communicate with `mlx_lm` Python server
- **HTTP Server**: SwiftNIO or NIOHTTP1
- **Architecture**: Service-based (ModelService, APIService, UIService)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    UI Layer                        │
│              (SwiftUI Views)                       │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                 Service Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ ModelService │  │ APIService   │  │ConfigStore│ │
│  └──────┬───────┘  └──────┬───────┘  └───────────┘ │
│         │                 │                        │
└─────────┼─────────────────┼──────────────────────┘
          │                 │
┌─────────▼─────────────────▼──────────────────────┐
│              MLX Backend Layer                      │
│  ┌──────────────────┐  ┌────────────────────────┐ │
│  │  Python Server   │  │   Swift MLX Bindings    │ │
│  │  (mlx_lm server) │  │   (if available)       │ │
│  └──────────────────┘  └────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

## API Reference

### GET /v1/models

Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
      "object": "model",
      "created": 1735689600,
      "owned_by": "mlx-community",
      "permission": [],
      "root": "/path/to/models",
      "parent": null
    }
  ]
}
```

### POST /v1/chat/completions

Request:
```json
{
  "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "stream": false
}
```

Response:
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1735689600,
  "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

## Configuration

Stored in `~/.mlx-provider/config.json`:

```json
{
  "modelDirectory": "~/Models/mlx",
  "apiPort": 8080,
  "apiKey": "optional-secret-key",
  "defaultModel": null,
  "maxTokens": 2048,
  "temperature": 0.7
}
```

## Supported Models

MLX-provider supports all models compatible with `mlx-lm`:
- Llama 3.2, 3.3
- Mistral 7B
- Qwen 2.5
- Gemma 2B, 7B
- Phi-4
- And thousands more from Hugging Face MLX Community

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 15.0+
- MLX framework
- Python 3.10+ (for mlx_lm backend)
- Xcode 16+
