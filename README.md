# MLX-provider

**OpenAI-compatible REST API server for MLX models on Apple Silicon**

MLX-provider is a native macOS desktop application that provides an OpenAI-compatible API for running MLX models locally. It combines a SwiftUI interface with a Python `mlx_lm` backend for inference on Apple Silicon GPUs.

## Features

- **Native macOS App** — SwiftUI interface with native look and feel
- **OpenAI-Compatible API** — Works with any OpenAI client (SDK, curl, etc.)
- **Model Management** — Scan, load, and unload models via UI or API
- **Streaming Support** — Real-time token streaming via SSE
- **Configurable** — Custom port, model directory, generation parameters

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 15.0+
- Python 3.10+ with `mlx_lm` package
- Xcode 16+ (for building from source)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MLX-provider App                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ SwiftUI UI  │  │ HTTP Server │  │  Python mlx_lm      │ │
│  │             │→│ (Swift/NIO) │→│  Backend Process     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

The app consists of three layers:

1. **SwiftUI Frontend** — Settings, model selection, server control
2. **Swift HTTP Server** — OpenAI-compatible REST API
3. **Python Backend** — `mlx_lm` for actual model inference

## Initial Setup

### 1. Install Python Dependencies

```bash
# Create a virtual environment (recommended)
python3 -m venv ~/.mlx-provider-venv
source ~/.mlx-provider-venv/bin/activate

# Install mlx_lm
pip install mlx_lm
```

Verify installation:
```bash
python -c "from mlx_lm import load, generate; print('mlx_lm OK')"
```

### 2. Download Models

**Option A: Via HuggingFace (automatic)**

Models are downloaded automatically when accessed via API. Popular models:
- `mlx-community/Llama-3.2-3B-Instruct-4bit` (~2GB)
- `mlx-community/Mistral-7B-Instruct-v0.3-4bit` (~4GB)
- `mlx-community/Qwen2.5-7B-Instruct-4bit` (~5GB)

**Option B: Local Models**

Place MLX models in a directory:

```
~/Models/mlx/
├── Llama-3.2-3B-Instruct-4bit/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
└── Mistral-7B/
    ├── config.json
    ├── model.safetensors
    └── ...
```

### 3. Build & Run

```bash
# Generate Xcode project
xcodegen generate

# Build
xcodebuild -project MLX-provider.xcodeproj -scheme MLX-provider -configuration Debug build

# Run (from Xcode or:)
open ~/Library/Developer/Xcode/DerivedData/*/Build/Products/Debug/MLX-provider.app
```

### 4. Configure

1. Launch MLX-provider
2. Click **Settings** (gear icon)
3. Set **Models Directory** to `~/Models/mlx` (or your preferred location)
4. Adjust **API Port** if needed (default: 8080)
5. Configure default generation parameters (max tokens, temperature)

## Usage

### Start the Server

1. Click **Start** in the main window
2. Status indicator turns green when running
3. API available at `http://localhost:8080/v1`

### API Endpoints

#### List Available Models

```bash
curl http://localhost:8080/v1/models
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

#### Streaming

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

#### Load/Unload Models

```bash
# Pre-load a model
curl -X POST http://localhost:8080/v1/models/{model_id}/load \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Llama-3.2-3B-Instruct-4bit"}'

# Unload a model
curl -X POST http://localhost:8080/v1/models/{model_id}/unload
```

### Client Integration

#### Python

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

#### JavaScript/TypeScript

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

## Configuration

Settings stored in `~/Library/Application Support/MLX-provider/config.json`:

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

| Setting | Description | Default |
|---------|-------------|---------|
| `modelDirectory` | Path to local models | `~/Models/mlx` |
| `apiPort` | HTTP server port | `8080` |
| `apiKey` | Optional API key authentication | `null` |
| `defaultModel` | Model to pre-load on startup | `null` |
| `maxTokens` | Default max tokens per response | `2048` |
| `temperature` | Default sampling temperature | `0.7` |

## Supported Models

MLX-provider supports all models compatible with `mlx_lm`:

- **Llama** 3.2, 3.3, 4.x
- **Mistral** 7B, 8x7B, 8x22B
- **Qwen** 2.5, 3.x
- **Gemma** 2B, 7B, 3B
- **Phi-4** and variants
- **OpenELM**
- **Thousands more** via HuggingFace MLX Community

## Troubleshooting

### "mlx_lm not installed"

```bash
pip install mlx_lm
```

### Server won't start

- Check if port 8080 is already in use: `lsof -i :8080`
- Verify Python path in app logs
- Ensure virtual environment is activated if using one

### Model won't load

- Check model directory path in Settings
- Verify model files exist and are valid MLX models
- For HuggingFace models, ensure internet connection

### Slow inference

- Use quantized models (4-bit, 8-bit) for better performance
- Larger models require more unified memory
- Close other memory-intensive applications

### Memory issues

- Use 4-bit quantized models to reduce memory footprint
- MLX uses unified memory — available RAM = used + GPU memory
- Monitor with Activity Monitor or `memory_pressure`

## Building from Source

### Prerequisites

```bash
# Install XcodeGen
brew install xcodegen
```

### Build

```bash
# Generate project
xcodegen generate

# Debug build
xcodebuild -project MLX-provider.xcodeproj -scheme MLX-provider -configuration Debug build

# Release build
xcodebuild -project MLX-provider.xcodeproj -scheme MLX-provider -configuration Release build
```

## License

MIT

## Links

- [MLX Framework](https://github.com/ml-explore/mlx)
- [MLX Swift](https://github.com/ml-explore/mlx-swift)
- [MLX Swift LM](https://github.com/ml-explore/mlx-swift-lm)
- [MLX Python](https://github.com/ml-explore/mlx-lm)
- [HuggingFace MLX Community](https://huggingface.co/mlx-community)
