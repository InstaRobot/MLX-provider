# MLX-provider

**Pure Swift OpenAI-compatible API server for MLX models on Apple Silicon**

MLX-provider is a desktop application that provides a local inference server for MLX (Apple's machine learning framework) models. It implements the OpenAI Chat Completions API format, allowing you to use Apple Silicon GPU acceleration with any OpenAI-compatible client. Built entirely in Swift — no Python required.

## Features

- 🖥️ **Native SwiftUI Desktop App** — macOS 15+ native interface
- ⚡ **Pure Swift MLX Backend** — No Python dependencies, direct MLX integration
- 🔌 **OpenAI-Compatible API** — Works with any OpenAI client library
- 📁 **Local + HuggingFace Models** — Scan local directories or use HF Hub models
- 🤖 **Dynamic Model Loading** — Load/unload models on demand
- 📊 **Streaming Support** — Real-time token streaming
- 🌐 **Local Server** — No internet required for inference

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 15.0+
- MLX Framework
- Xcode 16+

## Quick Start

### 1. Install MLX Swift Packages

The project uses Swift Package Manager. Open in Xcode:

```bash
cd MLX-provider
open project.yml
```

Or build from command line:

```bash
xcodegen generate
xcodebuild -project MLX-provider.xcodeproj -scheme MLX-provider -configuration Release build
```

### 2. Download Models

**Option A: HuggingFace MLX Community** (recommended for first use)

Models are downloaded automatically when you select them. Popular options:
- `mlx-community/Llama-3.2-3B-Instruct-4bit`
- `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
- `mlx-community/Qwen2.5-7B-Instruct-4bit`

**Option B: Local Models**

Place MLX models in a directory:

```
~/Models/mlx/
├── Llama-3.2-3B-Instruct-4bit/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
└── ...
```

### 3. Configure & Run

1. Launch MLX-provider
2. Open **Settings** (⚙️) and set your Models Directory
3. Click **Start** to launch the API server
4. The server runs at `http://localhost:8080`

## API Reference

### Base URL

```
http://localhost:8080/v1
```

### Endpoints

#### List Models

```bash
curl http://localhost:8080/v1/models
```

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
      "root": "/path/to/model"
    }
  ]
}
```

#### Chat Completions

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Apple Intelligence?"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
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
      "content": "Apple Intelligence is..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 48,
    "total_tokens": 60
  }
}
```

#### Load Model

```bash
curl -X POST http://localhost:8080/v1/models/{model_id}/load
```

#### Unload Model

```bash
curl -X POST http://localhost:8080/v1/models/{model_id}/unload
```

### Streaming

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

## Integration Examples

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8080/v1"
)

response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### JavaScript/TypeScript

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: 'not-needed',
  baseURL: 'http://localhost:8080/v1'
});

const response = await client.chat.completions.create({
  model: 'mlx-community/Llama-3.2-3B-Instruct-4bit',
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Hello!' }
  ]
});

console.log(response.choices[0].message.content);
```

### curl

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Your-Model-Name",
    "messages": [{"role": "user", "content": "Your question here"}]
  }'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        UI Layer                             │
│              (SwiftUI Desktop Application)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ContentView  │  │SettingsView │  │     AboutView       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────┬─────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   Service Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ ConfigStore  │  │ ModelService │  │   APIService    │  │
│  │ (Settings)   │  │(MLX Backend) │  │  (HTTP Server)  │  │
│  └──────────────┘  └──────┬───────┘  └────────┬────────┘  │
└───────────────────────────┼────────────────────┼────────────┘
                            │                    │
┌───────────────────────────▼────────────────────▼────────────┐
│                    MLX Framework Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐│
│  │   MLXLLM    │  │ MLXLMHugging│  │  MLXLMTokenizers   ││
│  │             │  │    Face     │  │                     ││
│  └─────────────┘  └─────────────┘  └─────────────────────┘│
│                                                             │
│  Apple Silicon GPU Acceleration ◄─────────────────────────►│
│  Neural Engine + GPU + Unified Memory                        │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

Configuration is stored in `~/Library/Application Support/MLX-provider/config.json`:

```json
{
  "modelDirectory": "~/Models/mlx",
  "apiPort": 8080,
  "apiKey": null,
  "defaultModel": null,
  "maxTokens": 2048,
  "temperature": 0.7
}
```

## Supported Models

MLX-provider supports all models from [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm):

- **Llama** 3.2, 3.3, 4.x
- **Mistral** 7B, 8x7B, 8x22B
- **Qwen** 2.5, 3.x
- **Gemma** 2B, 7B, 3B
- **Phi-4** and variants
- **OpenELM**
- **And thousands more** from HuggingFace MLX Community

## Building from Source

### Prerequisites

1. Install XcodeGen:
```bash
brew install xcodegen
```

2. Install MLX (if not already included):
```bash
# MLX will be installed via SPM
```

### Build

```bash
# Generate Xcode project
xcodegen generate

# Build
xcodebuild -project MLX-provider.xcodeproj -scheme MLX-provider -configuration Release build
```

### Run

```bash
# From Xcode, press Cmd+R
# Or from command line:
open build/Release/MLX-provider.app
```

## Troubleshooting

### Server won't start

- Check if port 8080 is already in use
- Verify your Models Directory exists and contains valid MLX models
- Check the Activity Log in the app for errors

### Model won't load

- Ensure you have macOS 15.0+ (required for MLX Swift)
- For HuggingFace models, check internet connection
- For local models, verify config.json and model files exist

### Slow inference

- Models large relative to RAM are slow on Apple Silicon
- Use quantized models (4-bit, 8-bit) for better performance
- Larger models like 70B require significant memory

### Memory issues

- Use 4-bit quantized models to reduce memory footprint
- Close other memory-intensive applications
- MLX uses unified memory — available RAM = used + GPU

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Links

- [MLX Framework](https://github.com/ml-explore/mlx)
- [MLX Swift](https://github.com/ml-explore/mlx-swift)
- [MLX Swift LM](https://github.com/ml-explore/mlx-swift-lm)
- [MLX Community Models](https://huggingface.co/mlx-community)
- [Documentation](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon)
