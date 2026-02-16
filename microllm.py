#!/usr/bin/env python3
"""microllm - Minimal LLM routing proxy.

Pure passthrough proxy that routes requests by model name to different backends.
No format translation - requests and responses are forwarded unchanged.
Drop-in replacement for LiteLLM with compatible YAML config.
"""

import asyncio
import json
import os
import sys
import time
from collections import defaultdict

import yaml
from aiohttp import web, ClientSession, ClientTimeout, TCPConnector


class MicroLLM:
    # Keep-alive settings for slow backends
    KEEPALIVE_INTERVAL = 30     # Send heartbeat every 30s if no data
    HARD_TIMEOUT = 600          # Hard timeout after 10 minutes
    DISCOVERY_COOLDOWN = 600    # Re-discover backends at most every 10 minutes

    def __init__(self, config_path, port=8012):
        self.port = port
        self.routes = {}        # model_name -> {api_base, model, api_key}
        self.ocr_url = None     # OCR service URL (e.g. http://localhost:8019)
        self.stats = defaultdict(lambda: {
            "requests": 0, "tokens_in": 0, "tokens_out": 0,
            "errors": 0, "total_gen_s": 0.0, "last_tok_s": 0.0,
        })
        self.start_time = time.time()
        self.last_discovery = 0.0  # timestamp of last discover_backends()
        self.session = None
        self.chatlog_dir = None
        self.chatlog_seq = 0
        self.load_config(config_path)

    def load_config(self, path):
        with open(path) as f:
            config = yaml.safe_load(f)

        settings = config.get("general_settings", {})
        self.port = settings.get("port", self.port)
        self.ocr_url = settings.get("ocr_url", None)
        if self.ocr_url:
            self.ocr_url = self.ocr_url.rstrip("/")
            print(f"microllm: OCR service at {self.ocr_url}")

        self.chatlog_dir = settings.get("chatlog_dir", None)
        if self.chatlog_dir:
            os.makedirs(self.chatlog_dir, exist_ok=True)
            print(f"microllm: chatlog -> {self.chatlog_dir}")

        for entry in config.get("model_list", []):
            name = entry["model_name"]
            params = entry.get("litellm_params", entry.get("params", {}))
            raw_model = params.get("model", name)
            # Strip provider prefix: "hosted_vllm/glm-4.7" -> "glm-4.7"
            model = raw_model.split("/", 1)[-1] if "/" in raw_model else raw_model
            api_base = params.get("api_base", "").rstrip("/")
            # Remove /v1 suffix if present (we append the full path ourselves)
            if api_base.endswith("/v1"):
                api_base = api_base[:-3]
            api_key = params.get("api_key", "dummy")
            self.routes[name] = {
                "api_base": api_base,
                "model": model,
                "api_key": api_key,
                "max_model_len": int(params.get("max_model_len", 0)),
            }

        print(f"microllm: {len(self.routes)} routes loaded:")
        for name, route in self.routes.items():
            print(f"  {name:20s} -> {route['api_base']}  (model: {route['model']})")

    # --- Request handlers ---

    async def handle_proxy(self, request):
        body_raw = await request.read()

        try:
            data = json.loads(body_raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return web.json_response(
                {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}},
                status=400,
            )

        # Intercept document blocks (PDF) -> OCR -> text
        if self.ocr_url and "messages" in data:
            data = await self._process_document_blocks(data)

        model_name = data.get("model", "")
        route = self.routes.get(model_name)
        if not route:
            return web.json_response(
                {
                    "error": {
                        "message": f"Unknown model: '{model_name}'. Available: {list(self.routes.keys())}",
                        "type": "invalid_request_error",
                    }
                },
                status=404,
            )

        # Replace model name with backend's expected name
        data["model"] = route["model"]

        body_out = json.dumps(data).encode()

        # Build backend URL (preserve full path + query string)
        path = request.path
        url = f"{route['api_base']}{path}"
        qs = request.query_string
        if qs:
            url += f"?{qs}"

        # Forward headers, skip hop-by-hop
        headers = {}
        for key, val in request.headers.items():
            if key.lower() not in ("host", "content-length", "transfer-encoding"):
                headers[key] = val
        headers["Content-Length"] = str(len(body_out))

        is_stream = data.get("stream", False)
        self.stats[model_name]["requests"] += 1
        self.chatlog_seq += 1
        req_seq = self.chatlog_seq
        t0 = time.monotonic()

        # Chatlog: write request
        if self.chatlog_dir:
            self._chatlog_write(req_seq, "req", data, model_name)

        try:
            async with self.session.post(url, data=body_out, headers=headers) as resp:
                content_type = resp.headers.get("content-type", "")

                if is_stream or "text/event-stream" in content_type:
                    return await self._stream_response(request, resp, model_name, t0, req_seq)
                else:
                    return await self._buffered_response(resp, model_name, t0, req_seq)
        except Exception as e:
            self.stats[model_name]["errors"] += 1
            return web.json_response(
                {"error": {"message": f"Backend error: {e}", "type": "proxy_error"}},
                status=502,
            )

    async def _stream_response(self, request, resp, model_name, t0, req_seq=0):
        """Forward SSE stream from backend to client with keep-alive heartbeats."""
        response = web.StreamResponse(
            status=resp.status,
            headers={
                "Content-Type": resp.headers.get("content-type", "text/event-stream"),
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        response.enable_chunked_encoding()
        await response.prepare(request)

        chunks_log = []
        heartbeats_sent = 0
        try:
            # Use async iteration with timeout for keep-alive
            reader = resp.content
            while True:
                elapsed_total = time.monotonic() - t0
                if elapsed_total > self.HARD_TIMEOUT:
                    print(f"  {model_name}: hard timeout after {elapsed_total:.0f}s", flush=True)
                    break

                try:
                    # Wait for data with timeout
                    chunk = await asyncio.wait_for(
                        reader.read(65536),
                        timeout=self.KEEPALIVE_INTERVAL
                    )
                    if not chunk:
                        break  # EOF
                    await response.write(chunk)
                    if self.chatlog_dir:
                        chunks_log.append(chunk)
                except asyncio.TimeoutError:
                    # No data within interval - send SSE keep-alive comment
                    heartbeats_sent += 1
                    keepalive = f": keepalive {heartbeats_sent} ({elapsed_total:.0f}s)\n\n".encode()
                    await response.write(keepalive)
                    print(f"  {model_name}: keepalive #{heartbeats_sent} at {elapsed_total:.0f}s", flush=True)
        except (ConnectionResetError, ConnectionError):
            pass  # Client disconnected
        finally:
            try:
                await response.write_eof()
            except Exception:
                pass
            elapsed = time.monotonic() - t0
            self.stats[model_name]["total_gen_s"] += elapsed
            status = f"stream" if heartbeats_sent == 0 else f"str+{heartbeats_sent}ka"
            self._log(model_name, status, elapsed)
            # Chatlog: write streamed response
            if self.chatlog_dir and chunks_log:
                self._chatlog_write_stream(req_seq, chunks_log, model_name)

        return response

    async def _buffered_response(self, resp, model_name, t0, req_seq=0):
        """Forward non-streaming response with keep-alive for slow backends."""
        # Read response body with keep-alive timeout handling
        chunks = []
        heartbeats_sent = 0
        reader = resp.content

        while True:
            elapsed_total = time.monotonic() - t0
            if elapsed_total > self.HARD_TIMEOUT:
                print(f"  {model_name}: hard timeout after {elapsed_total:.0f}s (buffered)", flush=True)
                return web.json_response(
                    {"error": {"message": f"Backend timeout after {elapsed_total:.0f}s", "type": "timeout_error"}},
                    status=504,
                )

            try:
                chunk = await asyncio.wait_for(
                    reader.read(65536),
                    timeout=self.KEEPALIVE_INTERVAL
                )
                if not chunk:
                    break  # EOF
                chunks.append(chunk)
            except asyncio.TimeoutError:
                heartbeats_sent += 1
                print(f"  {model_name}: waiting... #{heartbeats_sent} at {elapsed_total:.0f}s (buffered)", flush=True)
                # For buffered responses we can't send data yet, just log and continue waiting

        resp_body = b"".join(chunks)
        elapsed = time.monotonic() - t0

        # Try to extract token counts for stats
        try:
            resp_data = json.loads(resp_body)
            usage = resp_data.get("usage", {})
            tok_in = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
            tok_out = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
            self.stats[model_name]["tokens_in"] += tok_in
            self.stats[model_name]["tokens_out"] += tok_out
            self.stats[model_name]["total_gen_s"] += elapsed
            if tok_out and elapsed > 0:
                self.stats[model_name]["last_tok_s"] = round(tok_out / elapsed, 1)
        except (json.JSONDecodeError, AttributeError):
            pass

        status = f"{resp.status}" if heartbeats_sent == 0 else f"{resp.status}+{heartbeats_sent}w"
        self._log(model_name, status, elapsed)

        # Chatlog: write buffered response
        if self.chatlog_dir:
            try:
                self._chatlog_write(req_seq, "resp", json.loads(resp_body), model_name)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._chatlog_write(req_seq, "resp_raw", {"body": resp_body.decode("utf-8", errors="replace")[:2000]}, model_name)
        # Strip charset from content-type (aiohttp rejects it in content_type param)
        ct = resp.headers.get("content-type", "application/json")
        ct = ct.split(";")[0].strip()
        return web.Response(
            body=resp_body,
            status=resp.status,
            content_type=ct,
        )

    async def handle_health(self, request):
        return web.json_response({"status": "ok"})

    async def handle_models(self, request):
        # Lazy re-discovery: refresh if cooldown elapsed
        now = time.monotonic()
        if now - self.last_discovery > self.DISCOVERY_COOLDOWN:
            await self.discover_backends()

        models = [
            {
                "id": name,
                "object": "model",
                "backend": route["api_base"],
                "backend_model": route["model"],
            }
            for name, route in self.routes.items()
        ]
        return web.json_response({"object": "list", "data": models})

    async def handle_stats(self, request):
        uptime = time.time() - self.start_time
        models = {}
        for name in self.routes:
            s = dict(self.stats[name])
            # Compute average tok/s
            if s["tokens_out"] > 0 and s["total_gen_s"] > 0:
                s["avg_tok_s"] = round(s["tokens_out"] / s["total_gen_s"], 1)
            else:
                s["avg_tok_s"] = 0
            models[name] = s
        return web.json_response(
            {
                "uptime_s": round(uptime),
                "routes": len(self.routes),
                "models": models,
            },
            dumps=lambda x: json.dumps(x, indent=2),
        )

    # --- OCR Integration ---

    async def _process_document_blocks(self, data):
        """Replace type:'document' blocks with type:'text' via OCR service."""
        for msg in data.get("messages", []):
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            new_content = []
            for block in content:
                if (isinstance(block, dict)
                        and block.get("type") == "document"
                        and block.get("source", {}).get("media_type") == "application/pdf"):
                    pdf_b64 = block.get("source", {}).get("data", "")
                    if pdf_b64:
                        md = await self._call_ocr(pdf_b64)
                        new_content.append({
                            "type": "text",
                            "text": f"[PDF Document]\n\n{md}",
                        })
                    else:
                        new_content.append({
                            "type": "text",
                            "text": "[PDF Document: empty or unreadable]",
                        })
                else:
                    new_content.append(block)
            msg["content"] = new_content
        return data

    async def _call_ocr(self, pdf_base64):
        """Call OCR service to convert PDF to markdown."""
        try:
            async with self.session.post(
                f"{self.ocr_url}/ocr",
                json={"pdf_base64": pdf_base64},
                timeout=ClientTimeout(total=300),
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    md = result.get("markdown", "")
                    pages = result.get("pages", 0)
                    elapsed = result.get("elapsed_s", 0)
                    print(f"  OCR: {pages} pages in {elapsed}s", flush=True)
                    return md
                else:
                    body = await resp.text()
                    print(f"  OCR error {resp.status}: {body[:200]}", flush=True)
                    return f"[OCR error: {resp.status}]"
        except Exception as e:
            print(f"  OCR exception: {e}", flush=True)
            return f"[OCR unavailable: {e}]"

    # --- Chatlog ---

    def _chatlog_write(self, seq, kind, data, model_name):
        """Write a request or response to chatlog."""
        try:
            path = os.path.join(self.chatlog_dir, f"{seq:04d}_{kind}.json")
            with open(path, "w") as f:
                json.dump({"model": model_name, "time": time.strftime("%H:%M:%S"), "data": data}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  chatlog error: {e}", flush=True)

    def _chatlog_write_stream(self, seq, chunks, model_name):
        """Parse SSE chunks and write assembled response to chatlog."""
        try:
            raw = b"".join(chunks).decode("utf-8", errors="replace")
            events = []
            text_parts = []
            tool_blocks = []
            for line in raw.split("\n"):
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    continue
                try:
                    evt = json.loads(payload)
                    events.append(evt)
                    # Extract text deltas
                    etype = evt.get("type", "")
                    if etype == "content_block_start":
                        cb = evt.get("content_block", {})
                        if cb.get("type") == "tool_use":
                            tool_blocks.append({"id": cb.get("id"), "name": cb.get("name"), "input_json": ""})
                    elif etype == "content_block_delta":
                        delta = evt.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text_parts.append(delta.get("text", ""))
                        elif delta.get("type") == "input_json_delta":
                            if tool_blocks:
                                tool_blocks[-1]["input_json"] += delta.get("partial_json", "")
                except json.JSONDecodeError:
                    pass

            # Parse tool input JSON
            for tb in tool_blocks:
                try:
                    tb["input"] = json.loads(tb["input_json"])
                except (json.JSONDecodeError, KeyError):
                    tb["input"] = None  # PARSE FAILED
                del tb["input_json"]

            path = os.path.join(self.chatlog_dir, f"{seq:04d}_resp_stream.json")
            with open(path, "w") as f:
                json.dump({
                    "model": model_name,
                    "time": time.strftime("%H:%M:%S"),
                    "text": "".join(text_parts),
                    "tool_calls": tool_blocks,
                    "event_count": len(events),
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  chatlog stream error: {e}", flush=True)

    # --- Helpers ---

    def _log(self, model, status, elapsed):
        print(f"  {model:20s}  {status:6s}  {elapsed:7.1f}s  "
              f"reqs={self.stats[model]['requests']}  "
              f"in={self.stats[model]['tokens_in']}  "
              f"out={self.stats[model]['tokens_out']}", flush=True)

    # --- Server ---

    async def discover_backends(self):
        """Query backends to discover actual model names and create aliases."""
        self.last_discovery = time.monotonic()
        print("microllm: discovering backends...", flush=True)

        # Group routes by api_base to avoid duplicate queries
        base_to_routes = defaultdict(list)
        for name, route in self.routes.items():
            base_to_routes[route["api_base"]].append(name)

        new_aliases = {}  # model_name -> route_config (to add after iteration)

        for api_base, route_names in base_to_routes.items():
            try:
                url = f"{api_base}/v1/models"
                async with self.session.get(url, timeout=ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("data", [])
                        if models:
                            # Use first model as the actual backend model for configured routes
                            actual_model = models[0].get("id", "")
                            max_model_len = models[0].get("max_model_len", 0)
                            if actual_model:
                                for route_name in route_names:
                                    old_model = self.routes[route_name]["model"]
                                    self.routes[route_name]["model"] = actual_model
                                    if max_model_len:
                                        self.routes[route_name]["max_model_len"] = max_model_len
                                    if old_model != actual_model:
                                        print(f"  {route_name}: {old_model} -> {actual_model} (ctx:{max_model_len})", flush=True)
                                    else:
                                        print(f"  {route_name}: {actual_model} (ctx:{max_model_len})", flush=True)

                            # Add discovered real model names as aliases (cascade transparency)
                            # Only import where id == backend_model (skip routing aliases)
                            api_key = self.routes[route_names[0]]["api_key"]
                            for m in models:
                                mid = m.get("id", "")
                                mlen = m.get("max_model_len", 0)
                                backend_model = m.get("backend_model", mid)
                                if mid and mid == backend_model and mid not in self.routes and mid not in new_aliases:
                                    new_aliases[mid] = {
                                        "api_base": api_base,
                                        "model": mid,
                                        "api_key": api_key,
                                        "max_model_len": mlen,
                                    }
                                    print(f"  + alias: {mid} (ctx:{mlen})", flush=True)
                    else:
                        print(f"  {api_base}: HTTP {resp.status}", flush=True)
            except Exception as e:
                print(f"  {api_base}: {type(e).__name__} - {e}", flush=True)

        # Add discovered aliases
        self.routes.update(new_aliases)
        print(flush=True)

    async def run(self):
        connector = TCPConnector(
            limit=100,
            enable_cleanup_closed=True,
            force_close=False,
        )
        timeout = ClientTimeout(total=None, sock_read=600, sock_connect=10)
        self.session = ClientSession(timeout=timeout, connector=connector)

        # Auto-discover backend models
        await self.discover_backends()

        app = web.Application(client_max_size=100 * 1024 * 1024)  # 100 MB
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_get("/stats", self.handle_stats)
        app.router.add_route("*", "/{path:.*}", self.handle_proxy)

        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()

        print(f"microllm listening on 0.0.0.0:{self.port}", flush=True)
        print(f"  /health    - health check", flush=True)
        print(f"  /stats     - request statistics", flush=True)
        print(f"  /v1/models - list routes", flush=True)
        print(flush=True)

        try:
            await asyncio.Event().wait()
        finally:
            await self.session.close()
            await runner.cleanup()


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: microllm <config.yaml> [port]")
        print()
        print("Config format (LiteLLM-compatible):")
        print("  model_list:")
        print('    - model_name: local')
        print('      litellm_params:')
        print('        model: hosted_vllm/glm-4.7-flash')
        print('        api_base: http://localhost:8011/v1')
        sys.exit(0 if sys.argv[1:] == ["--help"] else 1)

    config_path = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8012

    proxy = MicroLLM(config_path, port)
    asyncio.run(proxy.run())


if __name__ == "__main__":
    main()
