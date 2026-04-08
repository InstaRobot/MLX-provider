#!/usr/bin/env python3
"""
MLX-provider Python Backend Server

This server wraps the mlx_lm package to provide an OpenAI-compatible API
for MLX models running on Apple Silicon.

Usage:
    python mlx_server.py --port 8080 --models-dir ~/Models/mlx
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Try to import mlx_lm
try:
    from mlx_lm import load, generate, stream_generate
    from mlx_lm.server import ModelProvider as MLXModelProvider
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False
    logger.warning("mlx_lm not installed. Run: pip install mlx_lm")


@dataclass
class ModelInfo:
    id: str
    name: str
    path: str
    size: int
    parameter_count: Optional[str] = None
    quantization: Optional[str] = None


@dataclass
class LoadedModel:
    model_id: str
    model: Any
    tokenizer: Any


class MLXServer:
    """MLX LLM Server - OpenAI compatible API"""
    
    def __init__(self, models_dir: str, host: str = "127.0.0.1", port: int = 8080):
        self.models_dir = Path(models_dir).expanduser()
        self.host = host
        self.port = port
        self.loaded_models: dict[str, LoadedModel] = {}
        self.default_model: Optional[str] = None
        
    def scan_models(self) -> list[ModelInfo]:
        """Scan models directory for available MLX models"""
        models = []
        
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return models
        
        for item in self.models_dir.iterdir():
            if item.is_dir():
                model_info = self._get_model_info(item)
                if model_info:
                    models.append(model_info)
            elif item.suffix in [".mlx", ".safetensors"]:
                # Single file model
                models.append(ModelInfo(
                    id=item.stem,
                    name=item.stem,
                    path=str(item),
                    size=item.stat().st_size
                ))
        
        return models
    
    def _get_model_info(self, path: Path) -> Optional[ModelInfo]:
        """Extract model info from directory"""
        config_path = path / "config.json"
        if not config_path.exists():
            return None
        
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            # Calculate approximate size
            total_size = sum(
                f.stat().st_size for f in path.rglob("*") if f.is_file()
            )
            
            # Try to extract parameter count
            param_count = None
            hidden_size = config.get("hidden_size")
            num_layers = config.get("num_hidden_layers")
            num_heads = config.get("num_attention_heads")
            intermediate_size = config.get("intermediate_size", hidden_size or 0)
            
            if hidden_size and num_layers and num_heads:
                # Approximate: 12 * hidden_size^2 for attention parameters
                approx = 12 * hidden_size * num_layers * (2 if intermediate_size == 0 else 1)
                if approx > 1e9:
                    param_count = f"{approx / 1e9:.1f}B"
                elif approx > 1e6:
                    param_count = f"{approx / 1e6:.1f}M"
            
            # Quantization info
            quant = None
            quant_config = config.get("quantization_config", {})
            if isinstance(quant_config, dict) and quant_config.get("bits"):
                quant = f"{quant_config['bits']}-bit"
            
            return ModelInfo(
                id=path.name,
                name=path.name,
                path=str(path),
                size=total_size,
                parameter_count=param_count,
                quantization=quant
            )
        except Exception as e:
            logger.warning(f"Failed to parse config for {path}: {e}")
            return None
    
    async def load_model(self, model_id: str) -> bool:
        """Load a model into memory"""
        if model_id in self.loaded_models:
            logger.info(f"Model {model_id} already loaded")
            return True
        
        models = self.scan_models()
        model_path = next((m.path for m in models if m.id == model_id), None)
        
        if not model_path:
            # Try as Hugging Face model ID
            model_path = model_id
        
        try:
            logger.info(f"Loading model: {model_path}")
            model, tokenizer = load(model_path)
            self.loaded_models[model_id] = LoadedModel(
                model_id=model_id,
                model=model,
                tokenizer=tokenizer
            )
            logger.info(f"Model {model_id} loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            traceback.print_exc()
            return False
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory"""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            logger.info(f"Model {model_id} unloaded")
            return True
        return False
    
    async def generate(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
        **kwargs
    ) -> dict:
        """Generate completion for chat messages"""
        if model_id not in self.loaded_models:
            raise ValueError(f"Model {model_id} not loaded")
        
        loaded = self.loaded_models[model_id]
        model, tokenizer = loaded.model, loaded.tokenizer
        
        # Apply chat template
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True
            )
        else:
            # Fallback to simple concatenation
            prompt = "\n".join(
                f"{m['role']}: {m['content']}" for m in messages
            )
        
        # Generate
        if stream:
            # Streaming response
            async def stream_response():
                for response in stream_generate(
                    model, tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                ):
                    yield {
                        "id": f"chatcmpl-{id(response)}",
                        "object": "chat.completion.chunk",
                        "created": int(__import__("time").time()),
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": response.text},
                            "finish_reason": None
                        }]
                    }
            return stream_response()
        else:
            # Non-streaming response
            text = generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
                **kwargs
            )
            
            return {
                "id": f"chatcmpl-{hash(prompt) % 1000000}",
                "object": "chat.completion",
                "created": int(__import__("time").time()),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(tokenizer.encode(prompt)),
                    "completion_tokens": len(tokenizer.encode(text)),
                    "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(text))
                }
            }


class APIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OpenAI-compatible API"""
    
    server: MLXServer
    
    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} - {format % args}")
    
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
    
    def do_GET(self):
        if self.path == "/v1/models":
            self.handle_list_models()
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        
        try:
            data = json.loads(body.decode()) if body else {}
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return
        
        if self.path == "/v1/chat/completions":
            self.handle_chat_completions(data)
        elif self.path.startswith("/v1/models/") and self.path.endswith("/load"):
            model_id = self.path.replace("/v1/models/", "").replace("/load", "")
            self.handle_load_model(model_id, data)
        elif self.path.startswith("/v1/models/") and self.path.endswith("/unload"):
            model_id = self.path.replace("/v1/models/", "").replace("/unload", "")
            self.handle_unload_model(model_id)
        else:
            self.send_error(404, "Not Found")
    
    def handle_list_models(self):
        models = self.server.scan_models()
        
        response = {
            "object": "list",
            "data": [
                {
                    "id": m.id,
                    "object": "model",
                    "created": int(__import__("time").time()),
                    "owned_by": "local",
                    "root": m.path,
                    "parent": None,
                    "info": {
                        "size": m.size,
                        "parameter_count": m.parameter_count,
                        "quantization": m.quantization
                    }
                } for m in models
            ]
        }
        
        self.send_json(response)
    
    def handle_chat_completions(self, data: dict):
        model_id = data.get("model", self.server.default_model)
        messages = data.get("messages", [])
        temperature = data.get("temperature", 0.7)
        max_tokens = data.get("max_tokens", 512)
        stream = data.get("stream", False)
        
        if not model_id:
            self.send_error(400, "Model not specified")
            return
        
        try:
            result = asyncio.run(
                self.server.generate(
                    model_id=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream
                )
            )
            
            if stream:
                self.send_stream_response(result)
            else:
                self.send_json(result)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            self.send_error(500, str(e))
    
    def handle_load_model(self, model_id: str, data: dict):
        target_model = data.get("model", model_id)
        
        success = asyncio.run(self.server.load_model(target_model))
        
        if success:
            self.send_json({"status": "loaded", "model": target_model})
        else:
            self.send_error(500, f"Failed to load model {target_model}")
    
    def handle_unload_model(self, model_id: str):
        success = asyncio.run(self.server.unload_model(model_id))
        
        if success:
            self.send_json({"status": "unloaded", "model": model_id})
        else:
            self.send_error(404, f"Model {model_id} not found")
    
    def send_json(self, data: dict, status: int = 200):
        response = json.dumps(data, ensure_ascii=False).encode()
        
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response)
    
    def send_stream_response(self, generator):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        
        for chunk in generator:
            data = json.dumps(chunk, ensure_ascii=False)
            self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()
        
        self.wfile.write(b"data: [DONE]\n\n")


def main():
    parser = argparse.ArgumentParser(description="MLX-provider Python Backend")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--models-dir", required=True, help="Directory containing MLX models")
    parser.add_argument("--default-model", help="Default model to load")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not MLX_LM_AVAILABLE:
        logger.error("mlx_lm is not installed. Please install it with: pip install mlx_lm")
        sys.exit(1)
    
    server = MLXServer(
        models_dir=args.models_dir,
        host=args.host,
        port=args.port
    )
    server.default_model = args.default_model
    
    # Pre-load default model if specified
    if args.default_model:
        logger.info(f"Pre-loading default model: {args.default_model}")
        asyncio.run(server.load_model(args.default_model))
    
    handler = lambda *args, **kwargs: APIHandler(*args, server=server, **kwargs)
    httpd = HTTPServer((args.host, args.port), handler)
    
    logger.info(f"MLX-provider server starting on {args.host}:{args.port}")
    logger.info(f"Models directory: {args.models_dir}")
    logger.info(f"API available at http://{args.host}:{args.port}/v1/chat/completions")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        httpd.shutdown()


if __name__ == "__main__":
    main()
