#!/usr/bin/env python3
"""microllm - Minimal LLM routing proxy.

Pure passthrough proxy that routes requests by model name to different backends.
No format translation - requests and responses are forwarded unchanged.
Drop-in replacement for LiteLLM with compatible YAML config.
"""

import asyncio
import json
import os
import signal
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

    HEALTH_CHECK_INTERVAL = 30     # Check unhealthy backends every 30s
    HEALTH_FAIL_THRESHOLD = 3      # Mark unhealthy after N consecutive failures
    HEALTH_COOLDOWN = 300          # Try unhealthy backends again after 5 minutes
    MAX_CONCURRENT_PER_BACKEND = 4 # Max parallel requests per backend

    def __init__(self, config_path, port=8012):
        self.config_path = config_path
        self.port = port
        self.routes = {}        # model_name -> [backend, ...]  (list for least-conn)
        self.services = {}      # service_name -> [backend, ...]  (generic HTTP proxy)
        self.rr_index = defaultdict(int)  # model_name -> next backend index
        self.backend_semaphores = {}  # api_base -> asyncio.Semaphore
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

    def get_backend_semaphore(self, api_base, max_concurrent=None):
        """Get or create a semaphore for a backend (limits concurrent requests)."""
        if api_base not in self.backend_semaphores:
            limit = max_concurrent or self.MAX_CONCURRENT_PER_BACKEND
            self.backend_semaphores[api_base] = asyncio.Semaphore(limit)
        return self.backend_semaphores[api_base]

    def load_config(self, path):
        with open(path) as f:
            config = yaml.safe_load(f)

        settings = config.get("general_settings", {})
        self.port = settings.get("port", self.port)
        self.ocr_url = settings.get("ocr_url", None)
        if self.ocr_url:
            self.ocr_url = self.ocr_url.rstrip("/")
            print(f"microllm: OCR service at {self.ocr_url}")

        self.MAX_CONCURRENT_PER_BACKEND = settings.get("max_concurrent_per_backend", 4)
        self.web_search_url = settings.get("web_search_url", None)
        if self.web_search_url:
            self.web_search_url = self.web_search_url.rstrip("/")
            print(f"microllm: web search at {self.web_search_url}")

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
            route = {
                "api_base": api_base,
                "model": model,
                "api_key": api_key,
                "max_model_len": int(params.get("max_model_len", 0)),
                "max_concurrent": int(params.get("max_concurrent", 0)),  # 0 = use global default
            }
            if "chat_template_kwargs" in params:
                route["chat_template_kwargs"] = params["chat_template_kwargs"]
            route["unhealthy_since"] = None
            route["fail_count"] = 0
            self.routes.setdefault(name, []).append(route)

        # Load service routes (generic HTTP proxy with round-robin)
        for entry in config.get("service_list", []):
            name = entry["service_name"]
            api_base = entry.get("api_base", "").rstrip("/")
            svc = {"api_base": api_base, "unhealthy_since": None, "fail_count": 0}
            self.services.setdefault(name, []).append(svc)

        print(f"microllm: {len(self.routes)} routes loaded:")
        for name, backends in self.routes.items():
            if len(backends) == 1:
                print(f"  {name:20s} -> {backends[0]['api_base']}  (model: {backends[0]['model']})")
            else:
                print(f"  {name:20s} -> {len(backends)} backends (least-conn):")
                for b in backends:
                    print(f"    - {b['api_base']}  (model: {b['model']})")
        if self.services:
            print(f"microllm: {len(self.services)} services loaded:")
            for name, backends in self.services.items():
                if len(backends) == 1:
                    print(f"  {name:20s} -> {backends[0]['api_base']}")
                else:
                    print(f"  {name:20s} -> {len(backends)} backends (least-conn):")
                    for b in backends:
                        print(f"    - {b['api_base']}")

    def reload_config(self):
        """Hot-reload config: merge new backends into existing groups, remove stale ones.
        In-flight requests keep their backend references and semaphores — safe to call anytime."""
        print(f"\nmicrollm: reloading {self.config_path}...", flush=True)
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"microllm: reload FAILED: {e}", flush=True)
            return f"reload failed: {e}"

        settings = config.get("general_settings", {})
        self.ocr_url = settings.get("ocr_url", None)
        if self.ocr_url:
            self.ocr_url = self.ocr_url.rstrip("/")
        self.MAX_CONCURRENT_PER_BACKEND = settings.get("max_concurrent_per_backend", 4)
        self.web_search_url = settings.get("web_search_url", None)
        if self.web_search_url:
            self.web_search_url = self.web_search_url.rstrip("/")
        self.chatlog_dir = settings.get("chatlog_dir", None)
        if self.chatlog_dir:
            os.makedirs(self.chatlog_dir, exist_ok=True)

        # Parse new model backends
        new_routes = {}
        for entry in config.get("model_list", []):
            name = entry["model_name"]
            params = entry.get("litellm_params", entry.get("params", {}))
            raw_model = params.get("model", name)
            model = raw_model.split("/", 1)[-1] if "/" in raw_model else raw_model
            api_base = params.get("api_base", "").rstrip("/")
            if api_base.endswith("/v1"):
                api_base = api_base[:-3]
            api_key = params.get("api_key", "dummy")
            route = {
                "api_base": api_base, "model": model, "api_key": api_key,
                "max_model_len": int(params.get("max_model_len", 0)),
                "max_concurrent": int(params.get("max_concurrent", 0)),
                "unhealthy_since": None, "fail_count": 0,
            }
            if "chat_template_kwargs" in params:
                route["chat_template_kwargs"] = params["chat_template_kwargs"]
            new_routes.setdefault(name, []).append(route)

        # Parse new service backends
        new_services = {}
        for entry in config.get("service_list", []):
            name = entry["service_name"]
            api_base = entry.get("api_base", "").rstrip("/")
            new_services.setdefault(name, []).append({
                "api_base": api_base, "unhealthy_since": None, "fail_count": 0,
            })

        added, removed = 0, 0

        # Merge model routes
        for name, new_backends in new_routes.items():
            if name not in self.routes:
                self.routes[name] = new_backends
                added += len(new_backends)
                print(f"  + new route: {name} ({len(new_backends)} backends)", flush=True)
                continue
            existing = self.routes[name]
            existing_bases = {b["api_base"] for b in existing}
            new_bases = {b["api_base"] for b in new_backends}
            # Add new backends
            for b in new_backends:
                if b["api_base"] not in existing_bases:
                    existing.append(b)
                    added += 1
                    print(f"  + {name}: added {b['api_base']}", flush=True)
            # Remove backends no longer in config (only if no in-flight requests)
            keep = []
            for b in existing:
                if b["api_base"] not in new_bases:
                    sem = self.backend_semaphores.get(b["api_base"])
                    limit = b.get("max_concurrent") or self.MAX_CONCURRENT_PER_BACKEND
                    in_flight = (limit - sem._value) if sem else 0
                    if in_flight > 0:
                        print(f"  ~ {name}: keeping {b['api_base']} (draining, {in_flight} in-flight)", flush=True)
                        keep.append(b)
                    else:
                        removed += 1
                        print(f"  - {name}: removed {b['api_base']}", flush=True)
                else:
                    keep.append(b)
            self.routes[name] = keep

        # Remove routes no longer in config (only if empty after backend removal)
        for name in list(self.routes.keys()):
            if name not in new_routes and not any(
                    b["api_base"] in self.backend_semaphores and
                    ((b.get("max_concurrent") or self.MAX_CONCURRENT_PER_BACKEND) -
                     self.backend_semaphores[b["api_base"]]._value) > 0
                    for b in self.routes[name]):
                del self.routes[name]
                print(f"  - removed route: {name}", flush=True)

        # Merge service routes (same logic)
        for name, new_backends in new_services.items():
            if name not in self.services:
                self.services[name] = new_backends
                added += len(new_backends)
                continue
            existing = self.services[name]
            existing_bases = {b["api_base"] for b in existing}
            new_bases = {b["api_base"] for b in new_backends}
            for b in new_backends:
                if b["api_base"] not in existing_bases:
                    existing.append(b)
                    added += 1
                    print(f"  + svc:{name}: added {b['api_base']}", flush=True)
            keep = []
            for b in existing:
                if b["api_base"] not in new_bases:
                    sem = self.backend_semaphores.get(b["api_base"])
                    in_flight = (self.MAX_CONCURRENT_PER_BACKEND - sem._value) if sem else 0
                    if in_flight > 0:
                        keep.append(b)
                    else:
                        removed += 1
                        print(f"  - svc:{name}: removed {b['api_base']}", flush=True)
                else:
                    keep.append(b)
            self.services[name] = keep

        msg = f"reload ok: +{added} -{removed} backends, {len(self.routes)} routes, {len(self.services)} services"
        print(f"microllm: {msg}\n", flush=True)
        return msg

    async def handle_reload(self, request):
        """HTTP endpoint for config reload."""
        msg = self.reload_config()
        return web.json_response({"status": msg})

    def _is_healthy(self, backend):
        """A backend is healthy if never failed, or if cooldown has elapsed."""
        us = backend.get("unhealthy_since")
        if us is None:
            return True
        return (time.monotonic() - us) >= self.HEALTH_COOLDOWN

    def pick_backend(self, model_name):
        """Pick healthy backend with most free slots (least-connections)."""
        backends = self.routes.get(model_name)
        if not backends:
            return None
        best, best_free = None, -1
        for b in backends:
            if not self._is_healthy(b):
                continue
            sem = self.backend_semaphores.get(b["api_base"])
            if sem is not None:
                free = sem._value
            else:
                free = b.get("max_concurrent") or self.MAX_CONCURRENT_PER_BACKEND
            if free > best_free:
                best, best_free = b, free
        if best:
            return best
        # All unhealthy — fall back to round-robin over unhealthy backends
        n = len(backends)
        start = self.rr_index[model_name] % n
        self.rr_index[model_name] = start + 1
        return backends[start]

    def mark_failed(self, model_name, backend):
        """Mark a backend as failed. After threshold, set unhealthy timestamp."""
        backend["fail_count"] += 1
        if backend["fail_count"] >= self.HEALTH_FAIL_THRESHOLD:
            if backend.get("unhealthy_since") is None:
                backend["unhealthy_since"] = time.monotonic()
                print(f"  {model_name}: backend {backend['api_base']} marked UNHEALTHY "
                      f"(after {backend['fail_count']} failures, retry in {self.HEALTH_COOLDOWN}s)",
                      flush=True)

    def mark_success(self, model_name, backend):
        """Mark a backend as successful. Reset fail count, clear unhealthy."""
        if backend.get("unhealthy_since") is not None:
            print(f"  {model_name}: backend {backend['api_base']} recovered", flush=True)
        backend["fail_count"] = 0
        backend["unhealthy_since"] = None

    # --- Request handlers ---

    async def handle_proxy(self, request):
        body_raw = await request.read()

        try:
            data = json.loads(body_raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Non-JSON request (e.g. multipart/form-data for audio uploads)
            # Extract model name from form data and forward as-is
            content_type = request.headers.get("content-type", "")
            if "multipart" in content_type or "/audio/" in request.path:
                return await self._passthrough_proxy(request, body_raw)
            return web.json_response(
                {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}},
                status=400,
            )

        # Intercept "Perform a web search" requests from Claude Code
        if self.web_search_url and "messages" in data:
            msgs = data.get("messages", [])
            if len(msgs) == 1 and msgs[0].get("role") == "user":
                content = msgs[0].get("content", "")
                search_prefix = "Perform a web search for the query:"
                # Handle both string and list content
                text = ""
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            text = block.get("text", "")
                            break
                if text.startswith(search_prefix):
                    query = text[len(search_prefix):].strip()
                    results = await self._web_search(query)
                    search_text = self._format_search_results(results, query)
                    msg_id = f"search-{int(time.time())}"
                    model = data.get("model", "web-search")

                    if data.get("stream"):
                        # Return SSE stream response
                        response = web.StreamResponse(
                            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
                        )
                        await response.prepare(request)
                        # message_start
                        await response.write(f'event: message_start\ndata: {json.dumps({"type":"message_start","message":{"id":msg_id,"type":"message","role":"assistant","content":[],"model":model,"stop_reason":None,"stop_sequence":None,"usage":{"input_tokens":0,"output_tokens":0}}})}\n\n'.encode())
                        # content_block_start
                        await response.write(f'event: content_block_start\ndata: {json.dumps({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})}\n\n'.encode())
                        # content_block_delta with full text
                        await response.write(f'event: content_block_delta\ndata: {json.dumps({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":search_text}})}\n\n'.encode())
                        # content_block_stop
                        await response.write(f'event: content_block_stop\ndata: {json.dumps({"type":"content_block_stop","index":0})}\n\n'.encode())
                        # message_delta
                        await response.write(f'event: message_delta\ndata: {json.dumps({"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":None},"usage":{"output_tokens":len(search_text.split())}})}\n\n'.encode())
                        # message_stop
                        await response.write(f'event: message_stop\ndata: {json.dumps({"type":"message_stop"})}\n\n'.encode())
                        await response.write_eof()
                        return response
                    else:
                        return web.json_response({
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "text", "text": search_text}],
                            "model": model,
                            "stop_reason": "end_turn",
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        })

        # Intercept document blocks (PDF) -> OCR -> text, or strip if no OCR
        if "messages" in data:
            data = await self._process_document_blocks(data)

        # Strip server-side tools that vLLM doesn't understand (web_search)
        # If web_search was requested AND we have a search engine configured,
        # perform the search and inject results into context
        if "tools" in data:
            has_web_search = False
            kept_tools = []
            for tool in data["tools"]:
                if tool.get("type", "").startswith("web_search"):
                    has_web_search = True
                else:
                    kept_tools.append(tool)
            data["tools"] = kept_tools
            if not data["tools"]:
                del data["tools"]

            # If web search was requested, search and inject results
            if has_web_search and self.web_search_url:
                query = self._extract_search_query(data)
                if query:
                    results = await self._web_search(query)
                    if results:
                        search_text = self._format_search_results(results, query)
                        # Inject as system context
                        system = data.get("system", "")
                        if isinstance(system, list):
                            system.append({"type": "text", "text": f"\n\n[Web Search Results]\n{search_text}"})
                        elif isinstance(system, str):
                            data["system"] = system + f"\n\n[Web Search Results]\n{search_text}"
                        else:
                            data["system"] = f"[Web Search Results]\n{search_text}"

        model_name = data.get("model", "")
        if model_name not in self.routes:
            return web.json_response(
                {
                    "error": {
                        "message": f"Unknown model: '{model_name}'. Available: {list(self.routes.keys())}",
                        "type": "invalid_request_error",
                    }
                },
                status=404,
            )

        is_stream = data.get("stream", False)
        client_ip = self._client_ip(request)
        self.stats[model_name]["requests"] += 1
        self.chatlog_seq += 1
        req_seq = self.chatlog_seq

        # Try backends with failover
        backends = self.routes[model_name]
        max_tries = len(backends)
        last_error = None

        for attempt in range(max_tries):
            route = self.pick_backend(model_name)
            if not route:
                break

            # Prepare request for this backend
            send_data = dict(data)
            send_data["model"] = route["model"]
            if "chat_template_kwargs" in route and "chat_template_kwargs" not in send_data:
                send_data["chat_template_kwargs"] = route["chat_template_kwargs"]

            body_out = json.dumps(send_data).encode()

            path = request.path
            url = f"{route['api_base']}{path}"
            qs = request.query_string
            if qs:
                url += f"?{qs}"

            headers = {}
            for key, val in request.headers.items():
                if key.lower() not in ("host", "content-length", "transfer-encoding"):
                    headers[key] = val
            headers["Content-Length"] = str(len(body_out))

            t0 = time.monotonic()

            if self.chatlog_dir and attempt == 0:
                self._chatlog_write(req_seq, "req", send_data, model_name)

            try:
                sem = self.get_backend_semaphore(route["api_base"], route.get("max_concurrent") or None)
                async with sem:
                    async with self.session.post(url, data=body_out, headers=headers) as resp:
                        content_type = resp.headers.get("content-type", "")
                        self.mark_success(model_name, route)

                        backend_id = route["api_base"]
                        if is_stream or "text/event-stream" in content_type:
                            return await self._stream_response(request, resp, model_name, t0, req_seq, client_ip, backend_id)
                        else:
                            return await self._buffered_response(resp, model_name, t0, req_seq, client_ip, backend_id)
            except Exception as e:
                self.mark_failed(model_name, route)
                last_error = e
                if max_tries > 1:
                    print(f"  {model_name}: backend {route['api_base']} failed ({e}), "
                          f"trying next ({attempt+1}/{max_tries})", flush=True)

        self.stats[model_name]["errors"] += 1
        return web.json_response(
            {"error": {"message": f"All backends failed: {last_error}", "type": "proxy_error"}},
            status=502,
        )

    async def _stream_response(self, request, resp, model_name, t0, req_seq=0, client_ip="-", backend_id="-"):
        """Forward SSE stream from backend to client with keep-alive heartbeats."""
        response = web.StreamResponse(
            status=resp.status,
            headers={
                "Content-Type": resp.headers.get("content-type", "text/event-stream"),
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Backend": backend_id,
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
            self._log(model_name, status, elapsed, client_ip)
            # Chatlog: write streamed response
            if self.chatlog_dir and chunks_log:
                self._chatlog_write_stream(req_seq, chunks_log, model_name)

        return response

    async def _buffered_response(self, resp, model_name, t0, req_seq=0, client_ip="-", backend_id="-"):
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
        self._log(model_name, status, elapsed, client_ip)

        # Chatlog: write buffered response
        if self.chatlog_dir:
            try:
                self._chatlog_write(req_seq, "resp", json.loads(resp_body), model_name)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._chatlog_write(req_seq, "resp_raw", {"body": resp_body.decode("utf-8", errors="replace")[:2000]}, model_name)
        # Strip charset from content-type (aiohttp rejects it in content_type param)
        ct = resp.headers.get("content-type", "application/json")
        ct = ct.split(";")[0].strip()
        response = web.Response(
            body=resp_body,
            status=resp.status,
            content_type=ct,
        )
        response.headers["X-Backend"] = backend_id
        return response

    # --- Passthrough proxy (multipart/audio, no JSON parsing) ---

    async def _passthrough_proxy(self, request, body_raw):
        """Forward non-JSON requests (multipart audio uploads) to backend by model name from form data."""
        # Try to extract model name from multipart form data
        model_name = None
        try:
            reader = await request.multipart()
            # Re-read body since multipart consumed it
            body_raw = await request.read()
        except Exception:
            pass

        # Parse model from form fields in raw body (simple search)
        content_type = request.headers.get("content-type", "")
        if not model_name:
            # Try form data field "model" from the raw multipart body
            import re
            match = re.search(rb'name="model"\r\n\r\n([^\r\n]+)', body_raw)
            if match:
                model_name = match.group(1).decode("utf-8", errors="replace").strip()

        if not model_name:
            # Fallback: try first route that handles audio
            for name in ("llm-stt", "whisper-large-v3"):
                if name in self.routes:
                    model_name = name
                    break

        if not model_name or model_name not in self.routes:
            return web.json_response(
                {"error": f"Cannot determine model for multipart request. Available: {list(self.routes.keys())}"},
                status=400,
            )

        route = self.pick_backend(model_name)
        if not route:
            return web.json_response({"error": f"No healthy backend for {model_name}"}, status=502)

        url = f"{route['api_base']}{request.path}"
        qs = request.query_string
        if qs:
            url += f"?{qs}"

        headers = {}
        for key, val in request.headers.items():
            if key.lower() not in ("host", "transfer-encoding"):
                headers[key] = val

        client_ip = self._client_ip(request)
        self.stats[model_name]["requests"] += 1
        t0 = time.monotonic()

        try:
            sem = self.get_backend_semaphore(route["api_base"])
            async with sem, self.session.request(
                request.method, url, data=body_raw, headers=headers,
                timeout=ClientTimeout(total=600),
            ) as resp:
                resp_body = await resp.read()
                elapsed = time.monotonic() - t0
                self.mark_success(model_name, route)
                self.stats[model_name]["total_gen_s"] += elapsed
                ct = resp.headers.get("content-type", "application/json").split(";")[0].strip()
                self._log(model_name, f"pt:{resp.status}", elapsed, client_ip)
                pt_resp = web.Response(body=resp_body, status=resp.status, content_type=ct)
                pt_resp.headers["X-Backend"] = route["api_base"]
                return pt_resp
        except Exception as e:
            self.mark_failed(model_name, route)
            self.stats[model_name]["errors"] += 1
            return web.json_response({"error": f"Backend failed: {e}"}, status=502)

    # --- Service proxy (generic HTTP, multipart, round-robin) ---

    def pick_service_backend(self, service_name):
        """Pick healthy service backend with most free slots (least-connections)."""
        backends = self.services.get(service_name)
        if not backends:
            return None
        best, best_free = None, -1
        for b in backends:
            if not self._is_healthy(b):
                continue
            sem = self.backend_semaphores.get(b["api_base"])
            if sem is not None:
                free = sem._value
            else:
                free = self.MAX_CONCURRENT_PER_BACKEND
            if free > best_free:
                best, best_free = b, free
        if best:
            return best
        # All unhealthy — fall back to round-robin
        n = len(backends)
        key = f"svc:{service_name}"
        start = self.rr_index[key] % n
        self.rr_index[key] = start + 1
        return backends[start]

    async def handle_service(self, request):
        """Proxy any request to a named service with round-robin + failover."""
        service_name = request.match_info["service"]
        sub_path = request.match_info.get("path", "")

        if service_name not in self.services:
            return web.json_response(
                {"error": f"Unknown service: '{service_name}'. Available: {list(self.services.keys())}"},
                status=404,
            )

        body = await request.read()
        client_ip = self._client_ip(request)
        backends = self.services[service_name]
        max_tries = len(backends)
        last_error = None
        svc_key = f"svc:{service_name}"
        self.stats[svc_key]["requests"] += 1

        for attempt in range(max_tries):
            backend = self.pick_service_backend(service_name)
            if not backend:
                break

            url = f"{backend['api_base']}/{sub_path}" if sub_path else backend["api_base"]
            qs = request.query_string
            if qs:
                url += f"?{qs}"

            headers = {}
            for key, val in request.headers.items():
                if key.lower() not in ("host", "transfer-encoding"):
                    headers[key] = val

            t0 = time.monotonic()
            try:
                sem = self.get_backend_semaphore(backend["api_base"])
                async with sem, self.session.request(
                    request.method, url, data=body, headers=headers,
                    timeout=ClientTimeout(total=600),
                ) as resp:
                    resp_body = await resp.read()
                    elapsed = time.monotonic() - t0
                    self.stats[svc_key]["total_gen_s"] += elapsed
                    backend["fail_count"] = 0
                    if backend.get("unhealthy_since") is not None:
                        backend["unhealthy_since"] = None
                        print(f"  svc:{service_name}: backend {backend['api_base']} recovered", flush=True)
                    ct = resp.headers.get("content-type", "application/octet-stream").split(";")[0].strip()
                    print(f"  {client_ip:15s}  svc:{service_name:15s}  {resp.status}  {elapsed:5.1f}s  "
                          f"{backend['api_base']}/{sub_path}", flush=True)
                    if resp.status >= 500 and attempt + 1 < max_tries:
                        backend["fail_count"] += 1
                        if backend["fail_count"] >= self.HEALTH_FAIL_THRESHOLD and backend.get("unhealthy_since") is None:
                            backend["unhealthy_since"] = time.monotonic()
                            print(f"  svc:{service_name}: backend {backend['api_base']} marked UNHEALTHY "
                                  f"(retry in {self.HEALTH_COOLDOWN}s)", flush=True)
                        print(f"  svc:{service_name}: backend {backend['api_base']} returned {resp.status}, "
                              f"trying next ({attempt+1}/{max_tries})", flush=True)
                        continue
                    return web.Response(body=resp_body, status=resp.status, content_type=ct)
            except Exception as e:
                backend["fail_count"] += 1
                if backend["fail_count"] >= self.HEALTH_FAIL_THRESHOLD and backend.get("unhealthy_since") is None:
                    backend["unhealthy_since"] = time.monotonic()
                    print(f"  svc:{service_name}: backend {backend['api_base']} marked UNHEALTHY "
                          f"(retry in {self.HEALTH_COOLDOWN}s)", flush=True)
                last_error = e
                if max_tries > 1:
                    print(f"  svc:{service_name}: backend {backend['api_base']} failed ({e}), "
                          f"trying next ({attempt+1}/{max_tries})", flush=True)

        self.stats[svc_key]["errors"] += 1
        return web.json_response(
            {"error": f"All backends failed: {last_error}"}, status=502,
        )

    async def handle_health(self, request):
        return web.json_response({"status": "ok"})

    async def handle_models(self, request):
        # Lazy re-discovery: refresh if cooldown elapsed
        now = time.monotonic()
        if now - self.last_discovery > self.DISCOVERY_COOLDOWN:
            await self.discover_backends()

        models = []
        for name, backends in self.routes.items():
            healthy = [b for b in backends if self._is_healthy(b)]
            models.append({
                "id": name,
                "object": "model",
                "backends": len(backends),
                "healthy": len(healthy),
                "backend": backends[0]["api_base"],
                "backend_model": backends[0]["model"],
            })
        return web.json_response({"object": "list", "data": models})

    async def handle_stats(self, request):
        uptime = time.time() - self.start_time
        models = {}
        for name, backends in self.routes.items():
            s = dict(self.stats[name])
            if s["tokens_out"] > 0 and s["total_gen_s"] > 0:
                s["avg_tok_s"] = round(s["tokens_out"] / s["total_gen_s"], 1)
            else:
                s["avg_tok_s"] = 0
            backend_list = []
            for b in backends:
                entry = {"api_base": b["api_base"], "healthy": self._is_healthy(b), "fail_count": b["fail_count"],
                         "unhealthy_since": b.get("unhealthy_since")}
                sem = self.backend_semaphores.get(b["api_base"])
                if sem is not None:
                    limit = b.get("max_concurrent") or self.MAX_CONCURRENT_PER_BACKEND
                    in_flight = limit - sem._value
                    entry["queue"] = {"in_flight": max(0, in_flight), "max": limit}
                backend_list.append(entry)
            s["backends"] = backend_list
            models[name] = s
        return web.json_response(
            {
                "uptime_s": round(uptime),
                "routes": len(self.routes),
                "models": models,
            },
            dumps=lambda x: json.dumps(x, indent=2),
        )

    async def handle_stats_reset(self, request):
        self.stats = defaultdict(lambda: {
            "requests": 0, "tokens_in": 0, "tokens_out": 0,
            "errors": 0, "total_gen_s": 0.0, "last_tok_s": 0.0,
        })
        self.start_time = time.time()
        return web.json_response({"status": "reset"})

    # --- OCR Integration ---

    async def _process_document_blocks(self, data):
        """Replace type:'document' blocks with type:'text' (via OCR or fallback)."""
        for msg in data.get("messages", []):
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            new_content = []
            for block in content:
                if (isinstance(block, dict)
                        and block.get("type") == "document"):
                    media_type = block.get("source", {}).get("media_type", "")
                    pdf_b64 = block.get("source", {}).get("data", "")
                    if pdf_b64 and self.ocr_url and "pdf" in media_type:
                        md = await self._call_ocr(pdf_b64)
                        new_content.append({
                            "type": "text",
                            "text": f"[PDF Document]\n\n{md}",
                        })
                    elif pdf_b64 and not self.ocr_url:
                        # No OCR available - insert placeholder
                        import base64
                        size_kb = len(pdf_b64) * 3 // 4 // 1024
                        new_content.append({
                            "type": "text",
                            "text": f"[Document: {media_type}, {size_kb}KB - content not extractable without OCR service]",
                        })
                    else:
                        new_content.append({
                            "type": "text",
                            "text": "[Document: empty or unreadable]",
                        })
                else:
                    new_content.append(block)
            msg["content"] = new_content
        return data

    async def _call_ocr(self, pdf_base64):
        """Extract text from PDF via Tika (PUT /tika/text with PDF body)."""
        import base64
        try:
            pdf_bytes = base64.b64decode(pdf_base64)
            url = self.ocr_url.rstrip('/') + '/tika/text'
            async with self.session.put(
                url,
                data=pdf_bytes,
                headers={"Content-Type": "application/pdf", "Accept": "application/json"},
                timeout=ClientTimeout(total=300),
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    text = result.get("X-TIKA:content", "").strip()
                    print(f"  Tika: extracted {len(text)} chars", flush=True)
                    return text
                else:
                    body = await resp.text()
                    print(f"  Tika error {resp.status}: {body[:200]}", flush=True)
                    return f"[Tika error: {resp.status}]"
        except Exception as e:
            print(f"  Tika exception: {e}", flush=True)
            return f"[Tika unavailable: {e}]"

    def _extract_search_query(self, data):
        """Extract a search query from the last user message."""
        msgs = data.get("messages", [])
        for m in reversed(msgs):
            if m.get("role") != "user":
                continue
            content = m.get("content", "")
            if isinstance(content, str):
                return content[:200]
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text and len(text) < 500:
                            return text
            break
        return None

    # --- Web Search ---

    async def _web_search(self, query, max_results=5):
        """Perform web search via configured search engine (SearXNG or DuckDuckGo)."""
        if not self.web_search_url:
            return []

        try:
            if self.web_search_url.lower() == "duckduckgo":
                return await self._search_duckduckgo(query, max_results)
            else:
                return await self._search_searxng(query, max_results)
        except Exception as e:
            print(f"  WebSearch error: {e}", flush=True)
            return [{"title": "Search error", "url": "", "snippet": str(e)}]

    async def _search_searxng(self, query, max_results):
        """Search via SearXNG JSON API."""
        params = f"?q={query}&format=json&engines=google,bing,duckduckgo&max_results={max_results}"
        url = f"{self.web_search_url}/search{params}"
        async with self.session.get(url, timeout=ClientTimeout(total=15)) as resp:
            if resp.status == 200:
                data = await resp.json()
                results = []
                for r in data.get("results", [])[:max_results]:
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("content", ""),
                    })
                print(f"  WebSearch: {len(results)} results for '{query[:50]}'", flush=True)
                return results
            else:
                print(f"  SearXNG error {resp.status}", flush=True)
                return []

    async def _search_duckduckgo(self, query, max_results):
        """Search via DuckDuckGo HTML (no API key needed)."""
        from urllib.parse import quote_plus
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        async with self.session.get(url, headers=headers, timeout=ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return []
            html = await resp.text()
            # Parse results from HTML
            results = []
            import re
            # DuckDuckGo HTML results are in <a class="result__a" href="...">title</a>
            # and <a class="result__snippet">snippet</a>
            links = re.findall(r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', html)
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
            for i, (href, title) in enumerate(links[:max_results]):
                snippet = snippets[i].strip() if i < len(snippets) else ""
                # Clean HTML tags from snippet/title
                title = re.sub(r'<[^>]+>', '', title).strip()
                snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                # DuckDuckGo wraps URLs in redirect
                if "uddg=" in href:
                    from urllib.parse import unquote, parse_qs, urlparse
                    parsed = urlparse(href)
                    actual_url = parse_qs(parsed.query).get("uddg", [href])[0]
                    href = unquote(actual_url)
                results.append({"title": title, "url": href, "snippet": snippet})
            print(f"  WebSearch(DDG): {len(results)} results for '{query[:50]}'", flush=True)
            return results

    def _format_search_results(self, results, query):
        """Format search results as text for the model."""
        if not results:
            return f"No web search results found for: {query}"
        lines = [f"Web search results for: \"{query}\"\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. [{r['title']}]({r['url']})")
            if r['snippet']:
                lines.append(f"   {r['snippet']}")
            lines.append("")
        return "\n".join(lines)

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

    @staticmethod
    def _client_ip(request):
        """Extract client IP from request (X-Forwarded-For or peername)."""
        xff = request.headers.get("X-Forwarded-For")
        if xff:
            return xff.split(",")[0].strip()
        peername = request.transport.get_extra_info("peername")
        return peername[0] if peername else "?"

    def _log(self, model, status, elapsed, client_ip="-"):
        print(f"  {client_ip:15s}  {model:20s}  {status:6s}  {elapsed:7.1f}s  "
              f"reqs={self.stats[model]['requests']}  "
              f"in={self.stats[model]['tokens_in']}  "
              f"out={self.stats[model]['tokens_out']}", flush=True)

    # --- Server ---

    async def discover_backends(self):
        """Query backends to discover actual model names and update health."""
        self.last_discovery = time.monotonic()
        print("microllm: discovering backends...", flush=True)

        # Collect all unique api_bases across all backends
        seen_bases = set()
        base_info = {}  # api_base -> {actual_model, max_model_len, api_key}

        for name, backends in self.routes.items():
            for backend in backends:
                base = backend["api_base"]
                if base in seen_bases:
                    continue
                seen_bases.add(base)
                try:
                    base_stripped = base.rstrip('/')
                    url = f"{base_stripped}/models" if base_stripped.endswith('/v1') else f"{base_stripped}/v1/models"
                    async with self.session.get(url, timeout=ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            models = data.get("data", [])
                            if models:
                                base_info[base] = {
                                    "actual_model": models[0].get("id", ""),
                                    "max_model_len": models[0].get("max_model_len", 0),
                                    "models": models,
                                }
                                backend["unhealthy_since"] = None
                                backend["fail_count"] = 0
                                print(f"  {base}: {models[0].get('id','')} (ctx:{models[0].get('max_model_len',0)})", flush=True)
                        else:
                            print(f"  {base}: HTTP {resp.status}", flush=True)
                            backend["unhealthy_since"] = time.monotonic()
                except Exception as e:
                    print(f"  {base}: {type(e).__name__} - {e}", flush=True)
                    backend["unhealthy_since"] = time.monotonic()

        # Update backend model names from discovery
        for name, backends in self.routes.items():
            for backend in backends:
                info = base_info.get(backend["api_base"])
                if info and info["actual_model"]:
                    backend["model"] = info["actual_model"]
                    if info["max_model_len"]:
                        backend["max_model_len"] = info["max_model_len"]

        # Add discovered model names as single-backend aliases
        new_aliases = {}
        for base, info in base_info.items():
            api_key = "dummy"
            # Find api_key from any backend using this base
            for backends in self.routes.values():
                for b in backends:
                    if b["api_base"] == base:
                        api_key = b["api_key"]
                        break
            for m in info.get("models", []):
                mid = m.get("id", "")
                if mid and mid not in self.routes and mid not in new_aliases:
                    new_aliases[mid] = [{
                        "api_base": base,
                        "model": mid,
                        "api_key": api_key,
                        "max_model_len": m.get("max_model_len", 0),
                        "unhealthy_since": None,
                        "fail_count": 0,
                    }]
                    print(f"  + alias: {mid}", flush=True)

        self.routes.update(new_aliases)
        print(flush=True)

    async def _health_check_loop(self):
        """Periodically probe unhealthy backends whose cooldown has elapsed."""
        while True:
            await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)
            # Check LLM routes
            for name, backends in self.routes.items():
                for backend in backends:
                    if backend.get("unhealthy_since") is None:
                        continue
                    if not self._is_healthy(backend):
                        continue  # Cooldown not yet elapsed
                    try:
                        base = backend['api_base'].rstrip('/')
                        url = f"{base}/models" if base.endswith('/v1') else f"{base}/v1/models"
                        async with self.session.get(url, timeout=ClientTimeout(total=5)) as resp:
                            if resp.status == 200:
                                backend["unhealthy_since"] = None
                                backend["fail_count"] = 0
                                print(f"  {name}: backend {backend['api_base']} recovered (health check)", flush=True)
                    except Exception:
                        backend["unhealthy_since"] = time.monotonic()  # Reset cooldown
            # Check service backends
            for name, backends in self.services.items():
                for backend in backends:
                    if backend.get("unhealthy_since") is None:
                        continue
                    if not self._is_healthy(backend):
                        continue
                    try:
                        base = backend['api_base'].rstrip('/')
                        async with self.session.get(base, timeout=ClientTimeout(total=5)) as resp:
                            if resp.status < 500:
                                backend["unhealthy_since"] = None
                                backend["fail_count"] = 0
                                print(f"  svc:{name}: backend {backend['api_base']} recovered (health check)", flush=True)
                    except Exception:
                        backend["unhealthy_since"] = time.monotonic()

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

        # Background health checker for unhealthy backends
        asyncio.create_task(self._health_check_loop())

        app = web.Application(client_max_size=500 * 1024 * 1024)  # 500 MB
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_get("/stats", self.handle_stats)
        app.router.add_post("/stats/reset", self.handle_stats_reset)
        app.router.add_post("/reload", self.handle_reload)
        app.router.add_route("*", "/svc/{service}/{path:.*}", self.handle_service)
        app.router.add_route("*", "/{path:.*}", self.handle_proxy)

        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()

        # SIGHUP triggers config reload
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGHUP, self.reload_config)

        print(f"microllm listening on 0.0.0.0:{self.port}", flush=True)
        print(f"  /health    - health check", flush=True)
        print(f"  /stats     - request statistics", flush=True)
        print(f"  /reload    - hot-reload config (POST)", flush=True)
        print(f"  /v1/models - list routes", flush=True)
        print(f"  SIGHUP     - hot-reload config", flush=True)
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
