# MLX-provider

**OpenAI-compatible REST API server for MLX models on Apple Silicon**

MLX-provider is a desktop application that provides a local inference server for MLX (Apple's machine learning framework) models. It implements the OpenAI Chat Completions API format, allowing you to use Apple Silicon GPU acceleration with any OpenAI-compatible client.

## Features

- 🖥️ **Desktop UI** — SwiftUI-based macOS application
- 🔌 **OpenAI-compatible API** — Works with any OpenAI client library
- 📁 **Model Directory Scanning** — Automatically discovers MLX models in a folder
- ⚡ **Apple Silicon GPU Acceleration** — Full MLX performance on M1/M2/M3/M4
- 🔄 **Dynamic Model Loading** — Load/unload models on demand
- 🌐 **Local Server** — No internet required for inference

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 15.0+
- MLX framework
- Python 3.10+ (for mlx_lm backend)
- Xcode 16+

## Installation

### 1. Install MLX LM

```bash
pip install mlx_lm
```

### 2. Download Models

Place your MLX models in a directory, e.g.:

```
~/Models/mlx/
├── Llama-3.2-3B-Instruct-4bit/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
├── Mistral-7B-Instruct/
│   └── ...
└── ...
```

Or download models from the [MLX Community](https://huggingface.co/mlx-community):

```bash
# Example: Download Llama 3.2 3B Instruct (4-bit)
huggingface-cli download mlx-community/Llama-3.2-3B-Instruct-4bit --local-dir ~/Models/mlx/Llama-3.2-3B-Instruct-4bit
```

### 3. Build & Run

```bash
# Open in Xcode
open project.yml

# Or build from command line
xcodegen generate
xcodebuild -project MLX-provider.xcodeproj -scheme MLX-provider -configuration Release build
```

### 4. Configure

1. Launch MLX-provider
2. Click the gear icon (⚙️) to open Settings
3. Set your Models Directory path
4. Configure API Port (default: 8080)
5. Optionally set an API Key

### 5. Start Server

Click **Start** in the main window. The server will start on `http://localhost:8080`.

## API Reference

### List Available Models

```bash
curl http://localhost:8080/v1/models
```

Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "Llama-3.2-3B-Instruct-4bit",
      "object": "model",
      "created": 1735689600,
      "owned_by": "local",
      "root": "/Users/you/Models/mlx/Llama-3.2-3B-Instruct-4bit"
    }
  ]
}
```

### Chat Completions

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-3.2-3B-Instruct-4bit",
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
  "model": "Llama-3.2-3B-Instruct-4bit",
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

### Load Model

```bash
curl -X POST http://localhost:8080/v1/models/Llama-3.2-3B-Instruct-4bit/load
```

### Unload Model

```bash
curl -X POST http://localhost:8080/v1/models/Llama-3.2-3B-Instruct-4bit/unload
```

## Integration with OpenAI Clients

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8080/v1"
)

response = client.chat.completions.create(
    model="Llama-3.2-3B-Instruct-4bit",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### JavaScript

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: 'not-needed',
  baseURL: 'http://localhost:8080/v1'
});

const response = await client.chat.completions.create({
  model: 'Llama-3.2-3B-Instruct-4bit',
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
│                     (SwiftUI App)                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    Service Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ ConfigStore │  │ ModelService │  │   APIService     │   │
│  └──────────────┘  └──────────────┘  │  (SwiftNIO)     │   │
│                                       └──────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 Backend Layer                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │         mlx_server.py (Python / mlx_lm)                │  │
│  │         - Model loading/unloading                       │  │
│  │         - Tokenization & generation                     │  │
│  │         - OpenAI API format                             │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
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

MLX-provider supports all models compatible with [mlx-lm](https://github.com/ml-explore/mlx-lm):

- **Llama** 3.2, 3.3, 4.x
- **Mistral** 7B, 8x7B, 8x22B
- **Qwen** 2.5, 3.x
- **Gemma** 2B, 7B
- **Phi-4**
- **And thousands more** from Hugging Face MLX Community

## Troubleshooting

### Server won't start

- Check if port 8080 is already in use
- Verify your Models Directory exists and contains valid MLX models

### Model won't load

- Ensure you have `mlx_lm` installed: `pip install mlx_lm`
- Check the model directory structure (needs `config.json`)
- For very large models, increase wired memory: `sudo sysctl iogpu.wired_limit_mb=N`

### Slow inference

- Models large relative to RAM are slow on Apple Silicon
- Increase system wired memory limit
- Use quantized models (4-bit, 8-bit) for better performance

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Links

- [MLX Framework](https://github.com/ml-explore/mlx)
- [mlx-lm](https://github.com/ml-explore/mlx-lm)
- [MLX Community Models](https://huggingface.co/mlx-community)
