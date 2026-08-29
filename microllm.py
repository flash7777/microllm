#!/usr/bin/env python3
"""microllm - Minimal LLM routing proxy.

Pure passthrough proxy that routes requests by model name to different backends.
No format translation - requests and responses are forwarded unchanged.
Drop-in replacement for LiteLLM with compatible YAML config.
"""

import asyncio
import base64
import ipaddress
import json
import os
import re
import signal
import socket
import sys
import time
from collections import defaultdict
from html.parser import HTMLParser
from urllib.parse import urlparse, urljoin, quote, unquote

import yaml
from aiohttp import web, ClientSession, ClientTimeout, TCPConnector


class _HTMLTextExtractor(HTMLParser):
    """Convert HTML to readable plain text (skips script/style/svg/...)."""
    SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "iframe", "canvas", "head"}
    BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "table", "tr", "td", "th",
                  "section", "article", "header", "footer", "main", "nav", "aside",
                  "blockquote", "pre", "figure", "figcaption", "form", "fieldset", "dl", "dt", "dd"}
    HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip_depth = 0
        self._link_href = None

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        if tag in self.HEADING_TAGS:
            self.parts.append("## ")
        if tag == "a":
            for key, val in attrs:
                if key == "href" and val and val.lower().startswith(("http://", "https://")):
                    self._link_href = val

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            if self.skip_depth > 0:
                self.skip_depth -= 1
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        elif tag == "a" and self._link_href:
            self.parts.append(f" ({self._link_href})")
            self._link_href = None

    def handle_data(self, data):
        if self.skip_depth == 0 and data.strip():
            self.parts.append(data)


def _html_to_text(html):
    """HTML -> plain text: skip scripts, block tags as newlines, keep http links."""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    lines = [ln.rstrip() for ln in "".join(parser.parts).splitlines()]
    result, blank = [], 0
    for ln in lines:
        if ln.strip():
            result.append(re.sub(r"[ \t]+", " ", ln))
            blank = 0
        else:
            blank += 1
            if blank == 1 and result:
                result.append("")
    return "\n".join(result).strip()


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
        self.route_meta = {}    # model_name -> {"builtin": [...], "pdf": {...}}  (pro Alias-Gruppe)
        self.alias_rules = []   # [(compiled_regex, target_group), ...]  (model_match entries)
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

        self.web_fetch_url = settings.get("web_fetch_url", None)
        if self.web_fetch_url:
            self.web_fetch_url = self.web_fetch_url.rstrip("/")
            print(f"microllm: web fetch service at {self.web_fetch_url}")

        self.chatlog_dir = settings.get("chatlog_dir", None)
        if self.chatlog_dir:
            os.makedirs(self.chatlog_dir, exist_ok=True)
            print(f"microllm: chatlog -> {self.chatlog_dir}")

        # Pass 1: concrete backends (alias_of entries are resolved in pass 2)
        alias_entries = []
        for entry in config.get("model_list", []):
            if "alias_of" in entry:
                alias_entries.append(entry)
                continue
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
            self._apply_group_meta(name, entry)

        # Pass 2: alias entries — exact alias (model_name + alias_of) shares the
        # target group's backend list (health state + semaphores included);
        # regex alias (model_match + alias_of) routes unknown model names.
        self.alias_rules = []
        alias_names = set()
        for entry in alias_entries:
            target = entry["alias_of"]
            if "litellm_params" in entry or "params" in entry:
                print(f"  ! alias entry with litellm_params + alias_of -> skipped")
                continue
            name = entry.get("model_name")
            pattern = entry.get("model_match")
            if not name and not pattern or (name and pattern):
                print(f"  ! alias entry needs exactly one of model_name/model_match -> skipped")
                continue
            if target in alias_names:
                print(f"  ! alias target '{target}' is itself an alias (no chaining) -> skipped")
                continue
            if pattern:
                try:
                    compiled = re.compile(pattern)
                except re.error as e:
                    print(f"  ! model_match '{pattern}': bad regex ({e}) -> skipped")
                    continue
                if target not in self.routes:
                    print(f"  ! model_match '{pattern}' -> unknown group '{target}' -> skipped")
                    continue
                self.alias_rules.append((compiled, target))
                print(f"  ~ regex: {pattern} -> {target}")
            else:
                if name in self.routes:
                    print(f"  ! alias '{name}' overwrites existing route -> skipped")
                    continue
                if target not in self.routes:
                    print(f"  ! alias '{name}' -> unknown group '{target}' -> skipped")
                    continue
                self.routes[name] = self.routes[target]
                inherited = self.route_meta.get(target)
                if inherited is not None:
                    self.route_meta[name] = {"builtin": list(inherited["builtin"]), "pdf": dict(inherited["pdf"])}
                self._apply_group_meta(name, entry)
                alias_names.add(name)
                print(f"  ~ alias: {name} -> {target} ({len(self.routes[target])} backends)")

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

    @staticmethod
    def _default_pdf_meta():
        return {"enabled": True, "images": False, "vision": False, "vector": False, "dpi": 100, "max_image_pages": 8}

    def _apply_group_meta(self, name, entry, store=None, flush=False):
        """Merge builtin/pdf keys of a model_list entry into the group meta store (pro Alias-Gruppe)."""
        store = store if store is not None else self.route_meta
        if "builtin" not in entry and "pdf" not in entry:
            return
        if name not in store or store[name] is None:
            store[name] = {"builtin": [], "pdf": self._default_pdf_meta()}
        meta = store[name]
        if "builtin" in entry:
            builtin = entry["builtin"]
            if isinstance(builtin, str):
                builtin = [builtin]
            if meta["builtin"] and meta["builtin"] != builtin:
                print(f"  ! {name}: conflicting builtin lists, keeping {meta['builtin']}", flush=flush)
            else:
                meta["builtin"] = list(builtin)
        if "pdf" in entry:
            meta["pdf"].update(entry["pdf"])
        if meta["builtin"]:
            print(f"  {name:20s} builtin={meta['builtin']}  pdf={meta['pdf']}", flush=flush)

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
        self.web_fetch_url = settings.get("web_fetch_url", None)
        if self.web_fetch_url:
            self.web_fetch_url = self.web_fetch_url.rstrip("/")
        self.chatlog_dir = settings.get("chatlog_dir", None)
        if self.chatlog_dir:
            os.makedirs(self.chatlog_dir, exist_ok=True)

        # Parse new model backends (alias_of entries are resolved after merging)
        new_routes = {}
        new_aliases = {}       # model_name -> target group
        new_alias_rules = []   # (pattern, target group)
        for entry in config.get("model_list", []):
            if "alias_of" in entry:
                if "litellm_params" in entry or "params" in entry:
                    print(f"  ! alias entry with litellm_params + alias_of -> skipped", flush=True)
                    continue
                name = entry.get("model_name")
                pattern = entry.get("model_match")
                if not name and not pattern or (name and pattern):
                    print(f"  ! alias entry needs exactly one of model_name/model_match -> skipped", flush=True)
                    continue
                if pattern:
                    new_alias_rules.append((pattern, entry["alias_of"]))
                else:
                    new_aliases[name] = entry["alias_of"]
                continue
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

        # Parse group meta (builtin/pdf, pro Alias-Gruppe)
        new_meta = {}
        for entry in config.get("model_list", []):
            name = entry.get("model_name")
            if name and ("builtin" in entry or "pdf" in entry):
                self._apply_group_meta(name, entry, store=new_meta, flush=True)

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
        valid_names = set(new_routes) | set(new_aliases)
        for name in list(self.routes.keys()):
            if name not in valid_names and not any(
                    b["api_base"] in self.backend_semaphores and
                    ((b.get("max_concurrent") or self.MAX_CONCURRENT_PER_BACKEND) -
                     self.backend_semaphores[b["api_base"]]._value) > 0
                    for b in self.routes[name]):
                del self.routes[name]
                print(f"  - removed route: {name}", flush=True)

        # Group meta (builtin/pdf): new config is the source of truth
        for name in list(self.route_meta):
            if name not in new_meta:
                del self.route_meta[name]
        self.route_meta.update(new_meta)

        # Re-resolve alias routes: exact aliases share the (just merged) target
        # group's backend list; stale aliases are removed (drain-aware).
        for name, target in new_aliases.items():
            if name in new_routes:
                print(f"  ! alias '{name}' overwrites existing route -> skipped", flush=True)
                continue
            if target in new_aliases:
                print(f"  ! alias target '{target}' is itself an alias (no chaining) -> skipped", flush=True)
                continue
            if target in self.routes:
                self.routes[name] = self.routes[target]
                inherited = self.route_meta.get(target)
                if inherited is not None and name not in new_meta:
                    self.route_meta[name] = {"builtin": list(inherited["builtin"]), "pdf": dict(inherited["pdf"])}
                print(f"  ~ alias: {name} -> {target} ({len(self.routes[target])} backends)", flush=True)
            else:
                if name in self.routes and not any(
                        b["api_base"] in self.backend_semaphores and
                        ((b.get("max_concurrent") or self.MAX_CONCURRENT_PER_BACKEND) -
                         self.backend_semaphores[b["api_base"]]._value) > 0
                        for b in self.routes[name]):
                    del self.routes[name]
                    self.route_meta.pop(name, None)
                    print(f"  - removed alias: {name} (target {target} gone)", flush=True)

        # Rebuild regex alias rules
        self.alias_rules = []
        for pattern, target in new_alias_rules:
            try:
                compiled = re.compile(pattern)
            except re.error as e:
                print(f"  ! model_match '{pattern}': bad regex ({e}) -> skipped", flush=True)
                continue
            if target in new_aliases:
                print(f"  ! model_match '{pattern}' -> alias '{target}' (no chaining) -> skipped", flush=True)
                continue
            if target in self.routes:
                self.alias_rules.append((compiled, target))
                print(f"  ~ regex: {pattern} -> {target}", flush=True)
            else:
                print(f"  ! model_match '{pattern}' -> unknown group '{target}' -> skipped", flush=True)

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

        # Resolve the alias group first (needed for the group meta: builtin tools, PDF)
        model_name = data.get("model", "")
        if model_name not in self.routes:
            # Regex aliases (model_match): first matching rule wins
            for pattern, target in self.alias_rules:
                if pattern.search(model_name):
                    print(f"  {model_name} -> group {target} (regex '{pattern.pattern}')", flush=True)
                    model_name = target
                    break
            else:
                return web.json_response(
                    {
                        "error": {
                            "message": f"Unknown model: '{model_name}'. Available: {list(self.routes.keys())}",
                            "type": "invalid_request_error",
                        }
                    },
                    status=404,
                )

        group_meta = self.route_meta.get(model_name)
        has_builtin = bool(group_meta and group_meta.get("builtin"))

        # PDF blocks (document/input_file) -> Taki text + images, per alias group
        if "messages" in data:
            data = await self._convert_pdf_blocks(data, group_meta, request.path)

        # Strip server-side tools that vLLM doesn't understand (web_search)
        # If web_search was requested AND we have a search engine configured,
        # perform the search and inject results into context.
        # Skipped when the group has builtin tools: the tool loop takes over.
        if "tools" in data and not has_builtin:
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

        # Builtin tool executor: the alias group runs web_search/web_fetch
        # server-side.  Streaming clients get a streaming-first tool loop
        # (passthrough with tool_use interception), non-streaming clients
        # get the original buffered tool loop.
        if has_builtin and "messages" in data:
            if data.get("stream", False):
                return await self._run_streaming_tool_loop(
                    request, data, model_name, group_meta)
            return await self._run_tool_loop(request, data, model_name, group_meta)

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
        sse_buf = b""
        stream_text_parts = []
        stream_usage = None
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
                    # Token-Accounting: SSE-Events aus dem Stream parsen (ohne Netz-Overhead)
                    sse_buf += chunk
                    while b"\n\n" in sse_buf:
                        event_raw, sse_buf = sse_buf.split(b"\n\n", 1)
                        ev = self._parse_sse_event(event_raw)
                        if ev:
                            # OpenAI-Stil: choices[].delta.content
                            for ch in ev.get("choices", []):
                                txt = ch.get("delta", {}).get("content")
                                if txt:
                                    stream_text_parts.append(txt)
                            # Anthropic-Stil: content_block_delta
                            if ev.get("type") == "content_block_delta":
                                d = ev.get("delta", {})
                                if d.get("type") == "text_delta":
                                    t = d.get("text", "")
                                    if t:
                                        stream_text_parts.append(t)
                            # usage (letztes Chunk, OpenAI- oder Anthropic-Format)
                            u = ev.get("usage")
                            if u:
                                stream_usage = u
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
            # Token-Zählung für Streamed-Responses
            usage = stream_usage
            tok_in = 0
            tok_out = 0
            if usage:
                tok_in = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
                tok_out = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
            if tok_out == 0 and stream_text_parts:
                # Fallback: Output-Token aus dem gesammelten Text schätzen
                tok_out = max(1, len("".join(stream_text_parts).split()))
            if tok_in:
                self.stats[model_name]["tokens_in"] += tok_in
            if tok_out:
                self.stats[model_name]["tokens_out"] += tok_out
                if elapsed > 0:
                    self.stats[model_name]["last_tok_s"] = round(tok_out / elapsed, 1)
            status = f"stream" if heartbeats_sent == 0 else f"str+{heartbeats_sent}ka"
            self._log(model_name, status, elapsed, client_ip)
            # Chatlog: write streamed response
            if self.chatlog_dir and chunks_log:
                self._chatlog_write_stream(req_seq, chunks_log, model_name)

        return response

    def _parse_sse_event(self, event_bytes):
        """Ein SSE-Event (Bytes) in ein dict parsen. Liefert None, wenn unparsebar."""
        text = event_bytes.decode("utf-8", errors="replace")
        for ln in text.splitlines():
            ln = ln.strip()
            if ln.startswith("data:") and ln[5:].strip() not in ("", "[DONE]"):
                try:
                    return json.loads(ln[5:].strip())
                except (json.JSONDecodeError, ValueError):
                    continue
        return None

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

        client_ip = self._client_ip(request)
        self.stats[model_name]["requests"] += 1
        t0 = time.monotonic()

        # Try backends with failover (same pattern as the JSON handler)
        backends = self.routes[model_name]
        max_tries = len(backends)
        failed_ids = set()
        last_error = None

        for attempt in range(max_tries):
            candidates = [b for b in backends if b["api_base"] not in failed_ids]
            if not candidates:
                break
            # Prefer healthy backends with most free slots (least-connections).
            # If none healthy (all in cooldown), fall back to any candidate.
            healthy = [b for b in candidates if self._is_healthy(b)]
            pool = healthy if healthy else candidates
            best, best_free = None, -1
            for b in pool:
                sem = self.backend_semaphores.get(b["api_base"])
                free = sem._value if sem is not None else (b.get("max_concurrent") or self.MAX_CONCURRENT_PER_BACKEND)
                if free > best_free:
                    best, best_free = b, free
            route = best

            # Translate client model name to backend model name (same as JSON handler: send_data["model"] = route["model"])
            if "model" in route and route["model"] != model_name:
                body_raw = body_raw.replace(
                    (b'name="model"\r\n\r\n' + model_name.encode()),
                    (b'name="model"\r\n\r\n' + route["model"].encode()),
                    1,
                )

            url = f"{route['api_base']}{request.path}"
            qs = request.query_string
            if qs:
                url += f"?{qs}"

            headers = {}
            for key, val in request.headers.items():
                if key.lower() not in ("host", "transfer-encoding"):
                    headers[key] = val
            headers["Content-Length"] = str(len(body_raw))

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
                    self._log(model_name, f"pt:{resp.status}", elapsed, client_ip)
                    pt_resp = web.Response(body=resp_body, status=resp.status,
                                            content_type=resp.headers.get("content-type", "application/json").split(";")[0].strip())
                    pt_resp.headers["X-Backend"] = route["api_base"]
                    return pt_resp
            except Exception as e:
                self.mark_failed(model_name, route)
                failed_ids.add(route["api_base"])
                last_error = e
                if max_tries > 1:
                    print(f"  {model_name}: passthrough backend {route['api_base']} failed ({e}), "
                          f"trying next ({attempt+1}/{max_tries})", flush=True)

        self.stats[model_name]["errors"] += 1
        return web.json_response(
            {"error": f"All backends failed: {last_error}"}, status=502,
        )

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
        for pattern, target in self.alias_rules:
            backends = self.routes.get(target, [])
            healthy = [b for b in backends if self._is_healthy(b)]
            models.append({
                "id": f"{pattern.pattern} (regex -> {target})",
                "object": "model",
                "backends": len(backends),
                "healthy": len(healthy),
                "backend": backends[0]["api_base"] if backends else "?",
                "backend_model": backends[0]["model"] if backends else "?",
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

    # --- PDF Conversion (per alias group, anthropic + openai) ---

    async def _convert_pdf_blocks(self, data, group_meta, path):
        """Convert PDF blocks to text (+ image blocks) before they reach the backend.

        Anthropic: content block type 'document'  (source.data = base64 PDF)
        OpenAI:    content part  type 'input_file' (file.file_data = data URL)

        Enabled per alias group via the 'pdf' meta (default: enabled, no images).
        Without OCR service or with pdf.enabled=false a placeholder is inserted
        (backends like vLLM cannot consume raw PDF blocks).
        """
        is_anthropic = path.endswith("/v1/messages")
        for msg in data.get("messages", []):
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            new_content = []
            for block in content:
                if not isinstance(block, dict):
                    new_content.append(block)
                    continue
                if is_anthropic and block.get("type") == "document":
                    source = block.get("source", {}) or {}
                    media_type = source.get("media_type", "")
                    pdf_b64 = source.get("data", "")
                    if "pdf" in media_type and pdf_b64:
                        new_content.extend(
                            await self._pdf_to_blocks(pdf_b64, (group_meta or {}).get("pdf"), is_anthropic))
                        continue
                elif not is_anthropic and block.get("type") == "input_file":
                    file_part = block.get("file", {}) or {}
                    file_data = file_part.get("file_data", "")
                    if file_data.startswith("data:application/pdf"):
                        pdf_b64 = file_data.split(",", 1)[1] if "," in file_data else ""
                        if pdf_b64:
                            new_content.extend(await self._pdf_to_blocks(
                                pdf_b64, (group_meta or {}).get("pdf"), is_anthropic,
                                file_part.get("filename", "")))
                            continue
                new_content.append(block)
            msg["content"] = new_content
        return data

    async def _pdf_to_blocks(self, pdf_b64, pdf_meta, is_anthropic, filename=""):
        """Convert one base64 PDF to text + image blocks (format dependent)."""
        pdf_meta = pdf_meta or self._default_pdf_meta()
        label = filename or "document"
        if not self.ocr_url or not pdf_meta.get("enabled", True):
            size_kb = len(pdf_b64) * 3 // 4 // 1024
            return [self._text_block(
                f"[PDF {label}: {size_kb}KB - content not extractable "
                f"(PDF conversion disabled or no OCR service)]")]

        want_images = pdf_meta.get("images", False)
        # Vision: scan images go straight into the chat as image blocks,
        # the VLM reads them — Taki skips OCR (ocr=0), no text needed.
        vision = want_images and pdf_meta.get("vision", False)
        pdf_dpi = int(pdf_meta.get("dpi", 100))
        max_pages = int(pdf_meta.get("max_image_pages", 8))
        # max_pages applies to IMAGE processing only (Taki). Text is always
        # extracted for ALL pages — "bis Seite 8 von 70" must never happen.
        # describe=1 always (with images): Taki only describes EMBEDDED
        # figures, linking them to the surrounding page text.
        # vector=1: render vector-drawing clusters (diagrams, schematics)
        # that have no embedded raster equivalent.
        text, images, total_pages = await self._call_pdf2chat(
            pdf_b64,
            images_enabled=want_images,
            describe=want_images,
            ocr=not vision,
            vector=vision,
            dpi=pdf_dpi,
            max_pages=max_pages,
        )
        # Taki < 20260824-2145: max_pages truncated the TEXT as well
        # (marker "[PDF gekürzt: nur die ersten N von P Seiten verarbeitet]").
        # Re-extract text-only for the full document, keep the cached images.
        if total_pages and total_pages != max_pages and text:
            full_text, _full_images, _ = await self._call_pdf2chat(
                pdf_b64,
                images_enabled=False,
                describe=False,
                ocr=True,
                vector=False,
                dpi=pdf_dpi,
                max_pages=total_pages,
            )
            if len(full_text) > len(text):
                print(f"  Tika: full-text re-extract "
                      f"({len(full_text)} > {len(text)} chars, {total_pages} pages)",
                      flush=True)
                text = full_text
        blocks = []
        if text:
            blocks.append(self._text_block(f"[PDF {label}]\n\n{text}"))
        for img in images:
            if not isinstance(img, dict):
                continue
            if vision and img.get("b64"):
                blocks.append(self._image_block(img["b64"], img.get("mime", "image/png"), is_anthropic))
            elif img.get("description"):
                img_kind = img.get("kind", "embedded")
                label_kind = "Scan-Seite" if img_kind == "scan" else "Abbildung"
                blocks.append(self._text_block(
                    f"[{label_kind} Seite {img.get('page', '?')}: {img['description']}"))
        if not blocks:
            blocks.append(self._text_block(f"[PDF {label}: no content extracted]"))
        return blocks

    @staticmethod
    def _text_block(text):
        return {"type": "text", "text": text}

    @staticmethod
    def _image_block(b64, mime, is_anthropic):
        if is_anthropic:
            return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}

    async def _call_pdf2chat(self, pdf_base64, images_enabled, describe, dpi, max_pages,
                             ocr=True, vector=False):
        """Extract text (+ images) from PDF via Taki.

        Tries PUT {ocr_url}/tika/pdf2chat (page markers + rescued images);
        falls back to /tika/text (text only) on 404 (older Taki).
        ocr=False: Taki skips OCR of weak pages (vision clients read the
        scan images directly).
        vector=True: Taki renders vector-drawing clusters (diagrams,
        schematics) via PyMuPDF + pdftoppm crop.
        Returns (text, images, total_pages).
        """
        try:
            pdf_bytes = base64.b64decode(pdf_base64)
        except Exception as e:
            return f"[PDF decode error: {e}]", [], 0
        query = f"images={1 if images_enabled else 0}&dpi={dpi}&max_pages={max_pages}"
        if describe:
            query += "&describe=1"
        if not ocr:
            query += "&ocr=0"
        if vector:
            query += "&vector=1"
        url = f"{self.ocr_url}/tika/pdf2chat?{query}"
        try:
            async with self.session.put(
                url, data=pdf_bytes,
                headers={"Content-Type": "application/pdf", "Accept": "application/json"},
                timeout=ClientTimeout(total=600),
            ) as resp:
                if resp.status == 404:
                    print("  Tika: /tika/pdf2chat missing, falling back to /tika/text", flush=True)
                    return await self._call_ocr(pdf_base64), [], 0
                if resp.status == 200:
                    result = await resp.json()
                    text = (result.get("text") or "").strip()
                    images = result.get("images") or []
                    total_pages = result.get("pages", 0)
                    print(f"  Tika pdf2chat: {len(text)} chars, {len(images)} images, "
                          f"{total_pages} pages", flush=True)
                    return text, images, total_pages
                body = await resp.text()
                print(f"  Tika pdf2chat error {resp.status}: {body[:200]}", flush=True)
                return f"[Tika error: {resp.status}]", [], 0
        except Exception as e:
            print(f"  Tika pdf2chat exception: {e}", flush=True)
            return f"[Tika unavailable: {e}]", [], 0

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
        # brandis: google/duckduckgo sind von der Pod-Egress-IP gebannt (CSE-Rate-Limit, CAPTCHA/403).
        # qwant+mojeek liefern relevante deutsche Ergebnisse; bing/startpage/brave als Reserve.
        params = f"?q={query}&format=json&engines=qwant,mojeek,startpage,brave,bing&max_results={max_results}"
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

    # --- Builtin Tool Executor (per alias group: web_search, web_fetch) ---

    BUILTIN_TOOL_DEFS = {
        "uri_search": (
            "Search the web. Returns a numbered list of results with title, URL and snippet. "
            "For technical questions (hardware specs, connections, protocols), include "
            "'manual', 'specification', 'datasheet' or 'pdf' in the query.",
            {"type": "object", "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {"type": "integer", "description": "Maximum number of results (default 5)"},
            }, "required": ["query"]},
        ),
        "uri_fetch": (
            "Fetch a web page and return its content as text. PDF files are "
            "processed: the result contains the extracted text plus page figures. "
            "ALWAYS use uri_fetch to read a URL from uri_search results before "
            "answering — search snippets are not enough for technical details. "
            "If the fetched page is a product landing page without specs, "
            "try fetching the manual/datasheet URL instead.",
            {"type": "object", "properties": {
                "url": {"type": "string", "description": "Absolute URL to fetch (http/https)"},
            }, "required": ["url"]},
        ),
    }

    WEB_FETCH_MAX_REDIRECTS = 3
    WEB_FETCH_TIMEOUT = 15          # seconds per hop
    WEB_FETCH_SERVICE_TIMEOUT = 90  # open_fetch retries internally (3x + backoff)
    WEB_FETCH_MAX_BYTES = 500 * 1024
    WEB_FETCH_MAX_CHARS = 30000
    WEB_FETCH_PDF_MAX_BYTES = 20 * 1024 * 1024  # PDFs to the PDF pipeline (Tika cap)

    def _tool_definitions(self, is_anthropic, names):
        """Tool definitions in the wire format of the given API."""
        defs = []
        for name in names:
            if name not in self.BUILTIN_TOOL_DEFS:
                continue
            description, schema = self.BUILTIN_TOOL_DEFS[name]
            if is_anthropic:
                defs.append({"name": name, "description": description, "input_schema": schema})
            else:
                defs.append({
                    "type": "function",
                    "function": {"name": name, "description": description, "parameters": schema},
                })
        return defs

    async def _run_tool_loop(self, request, data, model_name, group_meta):
        """Execute builtin tools (web_search/web_fetch) server-side in a loop.

        LLM APIs have no session, so per request: inject the tool definitions,
        call the backend non-streaming, execute the tool calls microllm
        injected itself, append the results to the conversation, repeat until
        the model answers or max rounds is reached (last round runs without
        tools to force a final answer). Client-declared tools are passed
        through untouched and NOT executed: if the model calls one, the raw
        response goes back to the client. If the client wanted a stream, the
        buffered final answer is delivered as a synthesized SSE stream.
        """
        is_anthropic = request.path.endswith("/v1/messages")
        is_stream = data.get("stream", False)
        max_rounds = int(group_meta.get("max_tool_rounds", 3))

        # Anthropic server tools (web_search_*) would break vLLM: strip them
        if "tools" in data and is_anthropic:
            data["tools"] = [t for t in data["tools"]
                             if not (isinstance(t, dict) and str(t.get("type", "")).startswith("web_search"))]
            if not data["tools"]:
                del data["tools"]

        # Don't inject tools the client already declared itself
        client_tool_names = set()
        for t in data.get("tools") or []:
            if not isinstance(t, dict):
                continue
            if is_anthropic:
                client_tool_names.add(t.get("name"))
            else:
                client_tool_names.add((t.get("function") or {}).get("name"))
        builtin_names = [n for n in group_meta.get("builtin", [])
                         if n in self.BUILTIN_TOOL_DEFS and n not in client_tool_names]
        injected = self._tool_definitions(is_anthropic, builtin_names)
        if injected:
            data["tools"] = list(data.get("tools") or []) + injected

        client_ip = self._client_ip(request)
        t0 = time.monotonic()
        self.stats[model_name]["requests"] += 1
        self.chatlog_seq += 1
        req_seq = self.chatlog_seq

        # Prepare SSE response early so we can send keepalives during backend calls
        sse_response = None
        if is_stream:
            sse_response = web.StreamResponse(
                headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            sse_response.enable_chunked_encoding()
            await sse_response.prepare(request)

        final_resp = None
        last_error = None
        rounds_done = 0

        for round_no in range(1, max_rounds + 1):
            # Last round without tools -> force a final answer
            send_data = dict(data)
            send_data["stream"] = False
            if round_no < max_rounds:
                send_data["tools"] = list(data.get("tools") or [])
            else:
                send_data.pop("tools", None)

            self.chatlog_seq += 1
            round_seq = self.chatlog_seq
            self._chatlog_write(round_seq, f"req_r{round_no}", send_data, model_name)

            try:
                resp_json = await self._tool_loop_backend_call(
                    request, model_name, send_data, sse_response)
            except Exception as e:
                last_error = e
                break

            self._chatlog_write(round_seq, f"resp_r{round_no}", resp_json, model_name)
            rounds_done = round_no

            calls, had_other = self._extract_builtin_tool_calls(resp_json, is_anthropic, builtin_names)
            if not calls or had_other or round_no == max_rounds:
                # Final answer (last round is final even if the model still
                # tries to call tools) - or a client tool call the client must
                # handle itself
                final_resp = resp_json
                break

            print(f"  {model_name}: builtin tools round {round_no}/{max_rounds}: "
                  f"{[c['name'] for c in calls]}", flush=True)
            results = {}
            for call in calls:
                results[call["id"]] = await self._execute_builtin(
                    call, group_meta, is_anthropic)
            data = self._append_tool_round(data, resp_json, calls, results, is_anthropic)

        self.stats[model_name]["total_gen_s"] += time.monotonic() - t0

        if final_resp is None:
            self.stats[model_name]["errors"] += 1
            error_body = {"error": {"message": f"Tool loop failed after {rounds_done} rounds: {last_error}",
                                    "type": "proxy_error"}}
            if sse_response is not None:
                # SSE already prepared — send error as SSE event, then close
                try:
                    await self._sse_write(sse_response, "error", error_body)
                    await sse_response.write_eof()
                except Exception:
                    pass
                return sse_response
            return web.json_response(error_body, status=502)

        self._log(model_name, f"tools:{rounds_done}", time.monotonic() - t0, client_ip)
        return await self._deliver_tool_loop_response(
            request, final_resp, model_name, is_stream, is_anthropic, req_seq, sse_response)

    # ------------------------------------------------------------------
    # Option A: executed builtin tool calls are rendered back to the client as
    # <details type="tool_calls"> text in the content stream. Open WebUI renders
    # those with ToolCallDisplay. It is pure content — it does NOT trigger the
    # client's own tool-execution loop (which would re-invoke the LLM with the
    # builtin tools it cannot execute, ending in an empty re-call).
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_result_text(result, max_chars=2000):
        """Render a tool result for the trace as text. Block lists (fetched
        PDFs) become their text parts + an image counter — base64 never
        reaches the HTML attributes."""
        if isinstance(result, list):
            text_parts = [b.get("text", "") for b in result
                          if isinstance(b, dict) and b.get("type") == "text"]
            n_images = sum(1 for b in result
                           if isinstance(b, dict) and b.get("type") in ("image", "image_url"))
            text = "\n".join(text_parts)
            if n_images:
                text += f"\n[{n_images} images passed to the model]"
        else:
            text = result if isinstance(result, str) else str(result)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[... truncated ...]"
        return text

    @classmethod
    def _tool_trace_details_html(cls, entries, max_result_chars=2000):
        import html as _html
        parts = []
        for entry in entries:
            name = str(entry.get("name", ""))
            call_id = str(entry.get("id", ""))
            arguments = entry.get("input", {}) or {}
            result = entry.get("result", "")
            result_text = cls._tool_result_text(result, max_result_chars)
            args_attr = _html.escape(json.dumps(arguments, ensure_ascii=False), quote=True)
            res_attr = _html.escape(json.dumps(result_text, ensure_ascii=False), quote=True)
            parts.append(
                f'<details type="tool_calls" done="true" id="{_html.escape(call_id, quote=True)}" '
                f'name="{_html.escape(name, quote=True)}" arguments="{args_attr}" '
                f'result="{res_attr}" files="" embeds="[]">\n'
                f'<summary>Tool Executed</summary>\n</details>\n')
        return "".join(parts)

    async def _sse_emit_tool_trace(self, sse_response, model_name, entries,
                                   is_anthropic, next_index=0):
        """Emit executed tool calls as <details type=tool_calls> content.
        Returns the next free block index (advanced for Anthropic text blocks)."""
        if not entries:
            return next_index
        details = self._tool_trace_details_html(entries)
        if not details:
            return next_index
        try:
            if is_anthropic:
                await self._sse_write(sse_response, "content_block_start", {
                    "type": "content_block_start", "index": next_index,
                    "content_block": {"type": "text", "text": ""}})
                await self._sse_write(sse_response, "content_block_delta", {
                    "type": "content_block_delta", "index": next_index,
                    "delta": {"type": "text_delta", "text": details}})
                await self._sse_write(sse_response, "content_block_stop", {
                    "type": "content_block_stop", "index": next_index})
                return next_index + 1
            base = {"id": f"chatcmpl_tc_{int(time.time() * 1000)}",
                    "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": model_name}
            await self._sse_write(sse_response, None, {
                **base, "choices": [{"index": 0, "delta": {"content": details},
                                     "finish_reason": None}]})
            return next_index
        except (ConnectionResetError, ConnectionError):
            return next_index

    # ------------------------------------------------------------------
    # Streaming Tool Loop: first round streams through with tool_use
    # interception, follow-up rounds are buffered with keepalive.
    # ------------------------------------------------------------------

    async def _run_streaming_tool_loop(self, request, data, model_name, group_meta):
        """Streaming-aware tool loop: first round streams token-by-token,
        builtin tool_use blocks are intercepted and executed server-side,
        follow-up rounds are buffered (with keepalive on the open SSE stream).
        If the model calls no builtin tools (99% of requests), this is pure
        streaming passthrough with zero buffering overhead.
        """
        is_anthropic = request.path.endswith("/v1/messages")
        max_rounds = int(group_meta.get("max_tool_rounds", 3))

        # Strip Anthropic server-side tools (web_search_* type) that vLLM ignores
        if "tools" in data and is_anthropic:
            data["tools"] = [t for t in data["tools"]
                             if not (isinstance(t, dict)
                                     and str(t.get("type", "")).startswith("web_search"))]
            if not data["tools"]:
                del data["tools"]

        # Determine which builtin tools to inject (skip client-declared ones)
        client_tool_names = set()
        for t in data.get("tools") or []:
            if not isinstance(t, dict):
                continue
            if is_anthropic:
                client_tool_names.add(t.get("name"))
            else:
                client_tool_names.add((t.get("function") or {}).get("name"))
        builtin_names = [n for n in group_meta.get("builtin", [])
                         if n in self.BUILTIN_TOOL_DEFS and n not in client_tool_names]
        injected = self._tool_definitions(is_anthropic, builtin_names)
        if injected:
            data["tools"] = list(data.get("tools") or []) + injected

        client_ip = self._client_ip(request)
        t0 = time.monotonic()
        self.stats[model_name]["requests"] += 1
        self.chatlog_seq += 1
        req_seq = self.chatlog_seq

        # Prepare SSE response early (client sees headers immediately)
        sse_response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
                     "Connection": "keep-alive", "X-Accel-Buffering": "no"})
        sse_response.enable_chunked_encoding()
        await sse_response.prepare(request)

        # --- Phase 1: Stream first round, intercept builtin tool_use ---
        intercept = await self._stream_with_tool_intercept(
            request, data, model_name, sse_response, builtin_names, is_anthropic)

        if intercept is None:
            # No builtin tool calls — pure streaming passthrough completed
            elapsed = time.monotonic() - t0
            self.stats[model_name]["total_gen_s"] += elapsed
            self._log(model_name, "stream", elapsed, client_ip)
            return sse_response

        # --- Phase 2: Execute intercepted builtin tools ---
        tool_calls = intercept["tool_calls"]
        print(f"  {model_name}: streaming tool intercept: "
              f"{[c['name'] for c in tool_calls]}", flush=True)
        results = {}
        for call in tool_calls:
            results[call["id"]] = await self._execute_builtin(
                call, group_meta, is_anthropic)

        # Option A: show the executed tool calls to the client (as content)
        anthropic_next_index = intercept["forwarded_index"]
        trace_entries = [{"id": c["id"], "name": c["name"], "input": c["input"],
                          "result": results.get(c["id"], "")} for c in tool_calls]
        anthropic_next_index = await self._sse_emit_tool_trace(
            sse_response, model_name, trace_entries, is_anthropic, anthropic_next_index)

        # Rebuild conversation: assistant content + tool results
        if is_anthropic:
            fake_resp = {"content": intercept["assistant_content"],
                         "stop_reason": "tool_use"}
        else:
            fake_resp = {"choices": [{"message": {
                "content": intercept.get("openai_text") or None,
                "tool_calls": [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"],
                                  "arguments": json.dumps(
                                      tc["input"], ensure_ascii=False)}}
                    for tc in tool_calls
                ],
            }}]}
        data = self._append_tool_round(data, fake_resp, tool_calls, results, is_anthropic)

        # --- Phase 3: Follow-up rounds (non-streaming + keepalive) ---
        rounds_done = 1
        final_resp = None
        last_error = None

        for round_no in range(2, max_rounds + 1):
            send_data = dict(data)
            send_data["stream"] = False
            if round_no < max_rounds:
                send_data["tools"] = list(data.get("tools") or [])
            else:
                send_data.pop("tools", None)

            try:
                resp_json = await self._tool_loop_backend_call(
                    request, model_name, send_data, sse_response)
            except Exception as exc:
                last_error = exc
                break

            rounds_done = round_no
            calls, had_other = self._extract_builtin_tool_calls(
                resp_json, is_anthropic, builtin_names)
            if not calls or had_other or round_no == max_rounds:
                final_resp = resp_json
                break

            print(f"  {model_name}: builtin tools round {round_no}/{max_rounds}: "
                  f"{[c['name'] for c in calls]}", flush=True)
            results = {}
            for call in calls:
                results[call["id"]] = await self._execute_builtin(
                    call, group_meta, is_anthropic)
            # Option A: show this round's tool calls to the client (as content)
            trace_entries = [{"id": c["id"], "name": c["name"], "input": c["input"],
                              "result": results.get(c["id"], "")} for c in calls]
            anthropic_next_index = await self._sse_emit_tool_trace(
                sse_response, model_name, trace_entries, is_anthropic,
                anthropic_next_index)
            data = self._append_tool_round(data, resp_json, calls, results, is_anthropic)

        elapsed = time.monotonic() - t0
        self.stats[model_name]["total_gen_s"] += elapsed

        if final_resp is None:
            try:
                await self._sse_write(sse_response, "error", {
                    "error": {"message": f"Tool loop failed: {last_error}",
                              "type": "proxy_error"}})
                await sse_response.write_eof()
            except Exception:
                pass
            self._log(model_name, f"str+tools:{rounds_done}:err", elapsed, client_ip)
            return sse_response

        # --- Phase 4: Emit follow-up answer on the existing SSE stream ---
        next_index = anthropic_next_index
        try:
            if is_anthropic:
                for block in final_resp.get("content", []):
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "thinking":
                        await self._sse_write(sse_response, "content_block_start", {
                            "type": "content_block_start", "index": next_index,
                            "content_block": {"type": "thinking", "thinking": ""}})
                        await self._sse_write(sse_response, "content_block_delta", {
                            "type": "content_block_delta", "index": next_index,
                            "delta": {"type": "thinking_delta",
                                      "thinking": block.get("thinking", "")}})
                        await self._sse_write(sse_response, "content_block_stop", {
                            "type": "content_block_stop", "index": next_index})
                        next_index += 1
                    elif block_type == "text":
                        await self._sse_write(sse_response, "content_block_start", {
                            "type": "content_block_start", "index": next_index,
                            "content_block": {"type": "text", "text": ""}})
                        await self._sse_write(sse_response, "content_block_delta", {
                            "type": "content_block_delta", "index": next_index,
                            "delta": {"type": "text_delta",
                                      "text": block.get("text", "")}})
                        await self._sse_write(sse_response, "content_block_stop", {
                            "type": "content_block_stop", "index": next_index})
                        next_index += 1
                usage = final_resp.get("usage", {}) or {}
                await self._sse_write(sse_response, "message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": final_resp.get("stop_reason", "end_turn"),
                              "stop_sequence": None},
                    "usage": {"output_tokens": usage.get("output_tokens", 0)}})
                await self._sse_write(sse_response, "message_stop", {"type": "message_stop"})
            else:
                # OpenAI format: synthesize final chunks
                choice = (final_resp.get("choices") or [{}])[0]
                message = choice.get("message", {}) or {}
                text = message.get("content") or ""
                chunk_id = f"chatcmpl_stl_{int(time.time())}"
                created = int(time.time())
                model = final_resp.get("model", model_name)
                base = {"id": chunk_id, "object": "chat.completion.chunk",
                        "created": created, "model": model}
                if text:
                    await self._sse_write(sse_response, None, {
                        **base, "choices": [{"index": 0, "delta": {"content": text},
                                             "finish_reason": None}]})
                await self._sse_write(sse_response, None, {
                    **base, "choices": [{"index": 0, "delta": {},
                                         "finish_reason": choice.get("finish_reason") or "stop"}]})
                await sse_response.write(b"data: [DONE]\n\n")
        except (ConnectionResetError, ConnectionError):
            pass
        finally:
            try:
                await sse_response.write_eof()
            except Exception:
                pass

        self._log(model_name, f"str+tools:{rounds_done}", elapsed, client_ip)
        return sse_response

    async def _stream_with_tool_intercept(self, request, data, model_name,
                                          sse_response, builtin_names, is_anthropic):
        """Stream first round to client, intercepting builtin tool_use blocks.

        Returns None if the stream completed normally (no builtin tool calls).
        Returns dict with tool_calls, assistant_content, forwarded_index when
        builtin tools were called and need server-side execution.
        Non-Anthropic format is passed through without interception.
        """
        send_data = dict(data)
        send_data["stream"] = True

        # Pick backend
        route = self.pick_backend(model_name)
        if not route:
            return None
        prep = dict(send_data)
        prep["model"] = route["model"]
        if "chat_template_kwargs" in route and "chat_template_kwargs" not in prep:
            prep["chat_template_kwargs"] = route["chat_template_kwargs"]
        body_out = json.dumps(prep).encode()
        url = f"{route['api_base']}{request.path}"
        qs = request.query_string
        if qs:
            url += f"?{qs}"
        headers = {}
        for key, val in request.headers.items():
            if key.lower() not in ("host", "content-length", "transfer-encoding"):
                headers[key] = val
        headers["Content-Length"] = str(len(body_out))

        # Tracking state for Anthropic tool_use interception
        assistant_content = []       # all content blocks for conversation rebuild
        suppressed_calls = []        # intercepted builtin tool calls
        cur_block = None             # current block being assembled
        cur_suppressed = False       # is current block a suppressed builtin tool?
        cur_tool_json_parts = []     # accumulate input_json_delta for tool blocks
        forwarded_index = 0          # next client-visible block index
        had_client_tools = False
        stop_reason = None

        # Tracking state for OpenAI tool_call interception
        openai_text_parts = []           # accumulated text content
        openai_tool_calls = {}           # tc index -> {id, name, arguments_parts}
        openai_buffered_tc_events = []   # raw bytes to replay if not intercepting
        openai_intercepting = False      # set when we intercept at finish_reason

        sem = self.get_backend_semaphore(
            route["api_base"], route.get("max_concurrent") or None)
        try:
            async with sem:
                async with self.session.post(url, data=body_out, headers=headers) as resp:
                    if resp.status >= 400:
                        body = await resp.read()
                        await sse_response.write(body)
                        try:
                            await sse_response.write_eof()
                        except Exception:
                            pass
                        return None

                    self.mark_success(model_name, route)
                    sse_buf = b""
                    reader = resp.content

                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                reader.read(65536),
                                timeout=self.KEEPALIVE_INTERVAL)
                            if not chunk:
                                break
                        except asyncio.TimeoutError:
                            try:
                                await sse_response.write(b": keepalive\n\n")
                            except (ConnectionResetError, ConnectionError):
                                return None
                            continue

                        sse_buf += chunk
                        while b"\n\n" in sse_buf:
                            event_raw, sse_buf = sse_buf.split(b"\n\n", 1)
                            event_bytes = event_raw + b"\n\n"

                            ev = self._parse_sse_event(event_raw)
                            if ev is None:
                                # SSE comment or [DONE]
                                if openai_intercepting:
                                    continue  # suppress [DONE] during intercept
                                await sse_response.write(event_bytes)
                                continue

                            if openai_intercepting:
                                continue  # suppress post-intercept chunks

                            if not is_anthropic:
                                # --- OpenAI event handling ---
                                choices = ev.get("choices", [])
                                if not choices:
                                    # Usage or metadata chunk — forward
                                    await sse_response.write(event_bytes)
                                    continue

                                choice = choices[0]
                                delta = choice.get("delta", {})
                                finish_reason = choice.get("finish_reason")

                                # Text content — forward immediately
                                if delta.get("content") is not None:
                                    await sse_response.write(event_bytes)
                                    openai_text_parts.append(
                                        delta["content"])
                                    continue

                                # Tool call deltas — buffer
                                tc_list = delta.get("tool_calls")
                                if tc_list:
                                    for tc in tc_list:
                                        idx = tc.get("index", 0)
                                        if idx not in openai_tool_calls:
                                            openai_tool_calls[idx] = {
                                                "id": "",
                                                "name": "",
                                                "arguments_parts": []}
                                        entry = openai_tool_calls[idx]
                                        if tc.get("id"):
                                            entry["id"] = tc["id"]
                                        fn = tc.get("function") or {}
                                        if fn.get("name"):
                                            entry["name"] = fn["name"]
                                        if fn.get("arguments"):
                                            entry["arguments_parts"].append(
                                                fn["arguments"])
                                    openai_buffered_tc_events.append(
                                        event_bytes)
                                    continue

                                # Finish reason
                                if (finish_reason == "tool_calls"
                                        and openai_tool_calls):
                                    all_builtin = all(
                                        tc["name"] in builtin_names
                                        for tc in openai_tool_calls.values())
                                    if all_builtin:
                                        # Intercept all tool calls
                                        for tc in openai_tool_calls.values():
                                            raw_args = "".join(
                                                tc["arguments_parts"])
                                            try:
                                                args = (json.loads(raw_args)
                                                        if raw_args else {})
                                            except json.JSONDecodeError:
                                                args = {}
                                            suppressed_calls.append({
                                                "id": tc["id"],
                                                "name": tc["name"],
                                                "input": args})
                                        openai_intercepting = True
                                        continue
                                    else:
                                        # Some non-builtin — replay all
                                        for buf in openai_buffered_tc_events:
                                            await sse_response.write(buf)
                                        await sse_response.write(event_bytes)
                                        openai_tool_calls.clear()
                                        openai_buffered_tc_events.clear()
                                        continue

                                # Everything else (role, stop, etc.)
                                await sse_response.write(event_bytes)
                                continue

                            # --- Anthropic event handling ---
                            ev_type = ev.get("type", "")

                            if ev_type == "message_start":
                                await self._sse_write(
                                    sse_response, "message_start", ev)
                                continue

                            if ev_type == "content_block_start":
                                block = ev.get("content_block", {})
                                block_type = block.get("type", "")

                                if block_type == "tool_use":
                                    tool_name = block.get("name", "")
                                    tool_id = block.get("id", "")
                                    if tool_name in builtin_names:
                                        cur_suppressed = True
                                        cur_block = {"type": "tool_use",
                                                     "id": tool_id,
                                                     "name": tool_name,
                                                     "input": {}}
                                        cur_tool_json_parts = []
                                    else:
                                        cur_suppressed = False
                                        had_client_tools = True
                                        cur_block = {"type": "tool_use",
                                                     "id": tool_id,
                                                     "name": tool_name,
                                                     "input": {}}
                                        cur_tool_json_parts = []
                                        fwd = dict(ev)
                                        fwd["index"] = forwarded_index
                                        await self._sse_write(
                                            sse_response,
                                            "content_block_start", fwd)
                                        forwarded_index += 1
                                else:
                                    # thinking / text — always forward
                                    cur_suppressed = False
                                    cur_block = {"type": block_type}
                                    fwd = dict(ev)
                                    fwd["index"] = forwarded_index
                                    await self._sse_write(
                                        sse_response,
                                        "content_block_start", fwd)
                                    forwarded_index += 1
                                continue

                            if ev_type == "content_block_delta":
                                delta = ev.get("delta", {})
                                delta_type = delta.get("type", "")

                                if cur_suppressed:
                                    if delta_type == "input_json_delta":
                                        cur_tool_json_parts.append(
                                            delta.get("partial_json", ""))
                                    continue

                                # Forward with the client-visible index
                                fwd = dict(ev)
                                fwd["index"] = forwarded_index - 1
                                await self._sse_write(
                                    sse_response,
                                    "content_block_delta", fwd)

                                # Collect text for conversation rebuild
                                if cur_block is not None:
                                    if delta_type == "thinking_delta":
                                        cur_block["thinking"] = (
                                            cur_block.get("thinking", "")
                                            + delta.get("thinking", ""))
                                        sig = delta.get("signature")
                                        if sig:
                                            cur_block["signature"] = sig
                                    elif delta_type == "text_delta":
                                        cur_block["text"] = (
                                            cur_block.get("text", "")
                                            + delta.get("text", ""))
                                    elif delta_type == "input_json_delta":
                                        cur_tool_json_parts.append(
                                            delta.get("partial_json", ""))
                                continue

                            if ev_type == "content_block_stop":
                                if cur_suppressed:
                                    raw_json = "".join(cur_tool_json_parts)
                                    try:
                                        cur_block["input"] = (
                                            json.loads(raw_json)
                                            if raw_json else {})
                                    except json.JSONDecodeError:
                                        cur_block["input"] = {}
                                    suppressed_calls.append({
                                        "id": cur_block["id"],
                                        "name": cur_block["name"],
                                        "input": cur_block["input"]})
                                    assistant_content.append(cur_block)
                                else:
                                    fwd = dict(ev)
                                    fwd["index"] = forwarded_index - 1
                                    await self._sse_write(
                                        sse_response,
                                        "content_block_stop", fwd)
                                    # Finalize non-tool blocks
                                    if cur_block is not None:
                                        if cur_block["type"] != "tool_use":
                                            assistant_content.append(
                                                dict(cur_block))
                                        else:
                                            # Client tool_use — also collect
                                            raw_json = "".join(
                                                cur_tool_json_parts)
                                            try:
                                                cur_block["input"] = (
                                                    json.loads(raw_json)
                                                    if raw_json else {})
                                            except json.JSONDecodeError:
                                                cur_block["input"] = {}
                                            assistant_content.append(
                                                dict(cur_block))

                                cur_block = None
                                cur_suppressed = False
                                cur_tool_json_parts = []
                                continue

                            if ev_type == "message_delta":
                                delta = ev.get("delta", {})
                                stop_reason = delta.get("stop_reason")
                                if (suppressed_calls
                                        and stop_reason == "tool_use"):
                                    # Hold back — we continue after tool exec
                                    continue
                                await self._sse_write(
                                    sse_response, "message_delta", ev)
                                continue

                            if ev_type == "message_stop":
                                if suppressed_calls:
                                    continue  # hold back
                                await self._sse_write(
                                    sse_response, "message_stop", ev)
                                continue

                            # Unknown event — forward raw
                            await sse_response.write(event_bytes)

        except (ConnectionResetError, ConnectionError):
            return None
        except Exception as exc:
            self.mark_failed(model_name, route)
            print(f"  {model_name}: streaming tool intercept error: {exc}",
                  flush=True)
            return None

        if not suppressed_calls:
            # No builtin tool calls — streaming completed normally
            try:
                await sse_response.write_eof()
            except Exception:
                pass
            return None

        return {
            "tool_calls": suppressed_calls,
            "assistant_content": assistant_content,
            "forwarded_index": forwarded_index,
            "openai_text": "".join(openai_text_parts),
        }

    async def _backend_call_json(self, request, model_name, send_data):
        """Non-streaming backend call with failover. Returns the parsed JSON body."""
        backends = self.routes[model_name]
        last_error = None
        for attempt in range(len(backends)):
            route = self.pick_backend(model_name)
            if not route:
                break
            s = dict(send_data)
            s["model"] = route["model"]
            if "chat_template_kwargs" in route and "chat_template_kwargs" not in s:
                s["chat_template_kwargs"] = route["chat_template_kwargs"]
            body_out = json.dumps(s).encode()
            url = f"{route['api_base']}{request.path}"
            qs = request.query_string
            if qs:
                url += f"?{qs}"
            headers = {}
            for key, val in request.headers.items():
                if key.lower() not in ("host", "content-length", "transfer-encoding"):
                    headers[key] = val
            headers["Content-Length"] = str(len(body_out))
            try:
                sem = self.get_backend_semaphore(route["api_base"], route.get("max_concurrent") or None)
                async with sem:
                    async with self.session.post(url, data=body_out, headers=headers) as resp:
                        if resp.status >= 400:
                            body = await resp.text()
                            self.mark_failed(model_name, route)
                            last_error = f"HTTP {resp.status} from {route['api_base']}: {body[:200]}"
                            print(f"  {model_name}: {last_error}, "
                                  f"trying next ({attempt+1}/{len(backends)})", flush=True)
                            continue
                        self.mark_success(model_name, route)
                        resp_json = await resp.json(content_type=None)
                        usage = resp_json.get("usage") or {}
                        tok_in = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0) or 0
                        tok_out = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0) or 0
                        if tok_in:
                            self.stats[model_name]["tokens_in"] += tok_in
                        if tok_out:
                            self.stats[model_name]["tokens_out"] += tok_out
                        return resp_json
            except Exception as e:
                self.mark_failed(model_name, route)
                last_error = e
                if attempt + 1 < len(backends):
                    print(f"  {model_name}: backend {route['api_base']} failed ({e}), "
                          f"trying next ({attempt+1}/{len(backends)})", flush=True)
        raise RuntimeError(f"All backends failed: {last_error}")

    async def _tool_loop_backend_call(self, request, model_name, send_data, sse_response):
        """Backend call with SSE keepalive comments sent to the client during wait."""
        if sse_response is None:
            return await self._backend_call_json(request, model_name, send_data)

        result_holder = [None, None]  # [result, exception]

        async def do_call():
            try:
                result_holder[0] = await self._backend_call_json(request, model_name, send_data)
            except Exception as exc:
                result_holder[1] = exc

        task = asyncio.create_task(do_call())
        keepalive_count = 0
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=self.KEEPALIVE_INTERVAL)
            except asyncio.TimeoutError:
                keepalive_count += 1
                try:
                    await sse_response.write(f": keepalive {keepalive_count}\n\n".encode())
                except (ConnectionResetError, ConnectionError):
                    break

        await task  # propagate CancelledError if any
        if result_holder[1] is not None:
            raise result_holder[1]
        return result_holder[0]

    def _extract_builtin_tool_calls(self, resp_json, is_anthropic, builtin_names):
        """Extract the builtin tool calls from a backend response.

        Returns (calls, had_other): calls = [{"id", "name", "input"}, ...] for
        the builtin tools, had_other = model also called tools we did not
        inject (client tools the client must handle itself).
        """
        calls, had_other = [], False
        if is_anthropic:
            for block in resp_json.get("content", []):
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                name = block.get("name")
                if name in builtin_names:
                    calls.append({"id": block.get("id"), "name": name,
                                  "input": block.get("input", {}) or {}})
                else:
                    had_other = True
        else:
            for choice in resp_json.get("choices", []):
                for tc in (choice.get("message", {}) or {}).get("tool_calls", []) or []:
                    fn = tc.get("function", {}) or {}
                    name = fn.get("name")
                    try:
                        args = json.loads(fn.get("arguments", "") or "{}")
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    if name in builtin_names:
                        calls.append({"id": tc.get("id"), "name": name, "input": args})
                    else:
                        had_other = True
        return calls, had_other

    def _append_tool_round(self, data, resp_json, calls, results, is_anthropic):
        """Append the assistant tool round + tool results to the conversation."""
        if is_anthropic:
            content = [b for b in resp_json.get("content", [])
                       if isinstance(b, dict) and b.get("type") in ("text", "tool_use")]
            data["messages"].append({"role": "assistant", "content": content})
            data["messages"].append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": c["id"],
                 "content": results.get(c["id"], "")}
                for c in calls
            ]})
        else:
            message = (resp_json.get("choices") or [{}])[0].get("message", {}) or {}
            assistant_msg = {"role": "assistant", "content": message.get("content")}
            assistant_msg["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c["input"], ensure_ascii=False)}}
                for c in calls
            ]
            data["messages"].append(assistant_msg)
            for c in calls:
                data["messages"].append({"role": "tool", "tool_call_id": c["id"],
                                         "content": results.get(c["id"], "")})
        return data

    async def _execute_builtin(self, call, group_meta=None, is_anthropic=False):
        """Execute one builtin tool. Returns the result for the model: a text
        string, or (for fetched PDFs) a list of text/image content blocks in
        the wire format of the target API (vision models read the figures)."""
        name = call["name"]
        args = call["input"] or {}
        t0 = time.monotonic()
        try:
            if name == "uri_search":
                query = str(args.get("query", "")).strip()
                if not query:
                    return "Error: missing 'query' argument"
                try:
                    max_results = int(args.get("max_results", 5))
                except (TypeError, ValueError):
                    max_results = 5
                results = await self._web_search(query, max_results=max(1, min(max_results, 10)))
                text = self._format_search_results(results, query)
            elif name == "uri_fetch":
                url = str(args.get("url", "")).strip()
                if not url:
                    return "Error: missing 'url' argument"
                fetched = await self._web_fetch(url)
                if isinstance(fetched, dict):
                    text = await self._fetched_pdf_to_blocks(
                        fetched, url, group_meta, is_anthropic)
                else:
                    text = fetched
            else:
                return f"Error: unknown builtin tool '{name}'"
        except Exception as e:
            text = f"Error: {e}"
        if isinstance(text, list):
            n_images = sum(1 for b in text
                           if isinstance(b, dict) and b.get("type") in ("image", "image_url"))
            print(f"  builtin {name}: {time.monotonic() - t0:.1f}s, "
                  f"{len(text)} blocks, {n_images} images", flush=True)
        else:
            print(f"  builtin {name}: {time.monotonic() - t0:.1f}s, {len(text)} chars", flush=True)
        return text

    async def _fetched_pdf_to_blocks(self, pdf_info, url, group_meta, is_anthropic):
        """Route a fetched PDF (open_fetch kind="pdf") through the same PDF
        pipeline as chat uploads (Tika text + figures/vision)."""
        b64 = pdf_info.get("b64", "")
        if not b64:
            return f"Error: PDF too large to process ({pdf_info.get('bytes', '?')} bytes): {url}"
        path = urlparse(url).path
        filename = unquote(path.rsplit("/", 1)[-1]) if path else "document.pdf"
        size_kb = len(b64) * 3 // 4 // 1024
        blocks = [self._text_block(f"Fetched PDF {url} ({size_kb}KB):")]
        blocks.extend(await self._pdf_to_blocks(
            b64, (group_meta or {}).get("pdf"), is_anthropic, filename))
        return blocks

    @staticmethod
    def _url_is_safe(url):
        """SSRF guard: only public http(s) URLs, no private/loopback addresses."""
        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.hostname
        if not host:
            return False
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        except socket.gaierror:
            return False
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
                return False
        return True

    async def _web_fetch_via_service(self, url):
        """Fetch via the open_fetch service (bot-safe: TLS-impersonation, retry, cache)."""
        service_url = f"{self.web_fetch_url}/fetch?url={quote(url, safe='')}"
        timeout = ClientTimeout(total=self.WEB_FETCH_SERVICE_TIMEOUT)
        async with self.session.get(service_url, timeout=timeout) as resp:
            if resp.status >= 400:
                body = (await resp.text())[:200]
                raise RuntimeError(f"webfetch service HTTP {resp.status}: {body}")
            data = await resp.json()
        if not data.get("ok"):
            return f"Error: {data.get('error', 'fetch failed')} for {url}"
        if data.get("kind") == "pdf":
            # open_fetch delivers PDFs raw (base64) — the tool executor routes
            # them through the PDF pipeline like chat uploads do
            return {"kind": "pdf", "url": data.get("final_url", url),
                    "bytes": data.get("bytes", 0), "b64": data.get("b64", "")}
        text = data.get("text", "")
        if len(text) > self.WEB_FETCH_MAX_CHARS:
            text = text[:self.WEB_FETCH_MAX_CHARS] + "\n[... truncated ...]"
        return f"Content of {url}:\n\n{text}"

    async def _web_fetch(self, url):
        """Fetch a URL (SSRF-guarded, redirects re-checked).

        Returns the page text, or (for PDFs) a dict {"kind": "pdf", ...}
        that the tool executor routes through the PDF pipeline.
        """
        if self.web_fetch_url:
            try:
                return await self._web_fetch_via_service(url)
            except Exception as e:
                print(f"  uri_fetch: service failed ({e}), falling back to in-process",
                      flush=True)
        current = url
        for _ in range(self.WEB_FETCH_MAX_REDIRECTS + 1):
            if not self._url_is_safe(current):
                return (f"Error: URL not allowed (only public http/https URLs, "
                        f"no private/loopback addresses): {current}")
            headers = {
                "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                               "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
                "Accept": "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.8",
            }
            try:
                async with self.session.get(
                    current, headers=headers, allow_redirects=False,
                    timeout=ClientTimeout(total=self.WEB_FETCH_TIMEOUT),
                ) as resp:
                    if resp.status in (301, 302, 303, 307, 308):
                        location = resp.headers.get("Location", "")
                        if not location:
                            return f"Error: redirect without Location (HTTP {resp.status})"
                        current = urljoin(current, location)
                        continue
                    if resp.status >= 400:
                        return f"Error: HTTP {resp.status} for {current}"
                    content_type = resp.headers.get("content-type", "")
                    if "pdf" in content_type:
                        pdf_chunk = await resp.content.read(
                            self.WEB_FETCH_PDF_MAX_BYTES + 1)
                        if len(pdf_chunk) > self.WEB_FETCH_PDF_MAX_BYTES:
                            return (f"Error: PDF too large (> "
                                    f"{self.WEB_FETCH_PDF_MAX_BYTES // (1024 * 1024)}MB): "
                                    f"{current}")
                        return {"kind": "pdf", "url": current, "bytes": len(pdf_chunk),
                                "b64": base64.b64encode(pdf_chunk).decode("ascii")}
                    if "octet-stream" in content_type:
                        return f"Error: {current} is not a text page (content-type: {content_type})"
                    chunk = await resp.content.read(self.WEB_FETCH_MAX_BYTES + 1)
            except asyncio.TimeoutError:
                return f"Error: timeout fetching {current} (> {self.WEB_FETCH_TIMEOUT}s)"
            except Exception as e:
                return f"Error fetching {current}: {e}"
            if len(chunk) > self.WEB_FETCH_MAX_BYTES:
                return f"Error: page too large (> {self.WEB_FETCH_MAX_BYTES // 1024}KB): {current}"
            text = chunk.decode("utf-8", errors="replace")
            if "html" in content_type:
                text = _html_to_text(text)
            if len(text) > self.WEB_FETCH_MAX_CHARS:
                text = text[:self.WEB_FETCH_MAX_CHARS] + "\n[... truncated ...]"
            return f"Content of {url}:\n\n{text}"
        return f"Error: too many redirects for {url}"

    async def _deliver_tool_loop_response(self, request, final_resp, model_name,
                                          is_stream, is_anthropic, req_seq,
                                          sse_response=None):
        """Deliver the final tool-loop answer (JSON, or synthesized SSE if the client wanted a stream)."""
        if self.chatlog_dir:
            self._chatlog_write(req_seq, "resp", final_resp, model_name)
        if not is_stream:
            response = web.json_response(final_resp)
            response.headers["X-Backend"] = "tool-loop"
            return response

        # Re-use the pre-prepared SSE response (keepalives already flowing),
        # or create a new one if none was passed.
        if sse_response is not None:
            response = sse_response
        else:
            response = web.StreamResponse(
                headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
            )
            response.enable_chunked_encoding()
            await response.prepare(request)
        model = final_resp.get("model", model_name)
        try:
            if is_anthropic:
                msg_id = final_resp.get("id", f"msg_toolloop_{int(time.time())}")
                usage = final_resp.get("usage", {}) or {}
                await self._sse_write(response, "message_start", {
                    "type": "message_start",
                    "message": {"id": msg_id, "type": "message", "role": "assistant", "content": [],
                                "model": model, "stop_reason": None, "stop_sequence": None,
                                "usage": {"input_tokens": usage.get("input_tokens", 0),
                                          "output_tokens": 0}},
                })
                index = 0
                for block in final_resp.get("content", []):
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "thinking":
                        await self._sse_write(response, "content_block_start", {
                            "type": "content_block_start", "index": index,
                            "content_block": {"type": "thinking", "thinking": ""}})
                        await self._sse_write(response, "content_block_delta", {
                            "type": "content_block_delta", "index": index,
                            "delta": {"type": "thinking_delta",
                                      "thinking": block.get("thinking", "")}})
                        await self._sse_write(response, "content_block_stop", {
                            "type": "content_block_stop", "index": index})
                        index += 1
                    elif block_type == "text":
                        await self._sse_write(response, "content_block_start", {
                            "type": "content_block_start", "index": index,
                            "content_block": {"type": "text", "text": ""}})
                        await self._sse_write(response, "content_block_delta", {
                            "type": "content_block_delta", "index": index,
                            "delta": {"type": "text_delta", "text": block.get("text", "")}})
                        await self._sse_write(response, "content_block_stop", {
                            "type": "content_block_stop", "index": index})
                        index += 1
                await self._sse_write(response, "message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": final_resp.get("stop_reason", "end_turn"),
                              "stop_sequence": None},
                    "usage": {"output_tokens": usage.get("output_tokens", 0)},
                })
                await self._sse_write(response, "message_stop", {"type": "message_stop"})
            else:
                choice = (final_resp.get("choices") or [{}])[0]
                message = choice.get("message", {}) or {}
                text = message.get("content") or ""
                chunk_id = f"chatcmpl_toolloop_{int(time.time())}"
                created = int(time.time())
                base = {"id": chunk_id, "object": "chat.completion.chunk",
                        "created": created, "model": model}
                await self._sse_write(response, None, {
                    **base, "choices": [{"index": 0,
                                         "delta": {"role": "assistant", "content": ""},
                                         "finish_reason": None}]})
                if text:
                    await self._sse_write(response, None, {
                        **base, "choices": [{"index": 0, "delta": {"content": text},
                                             "finish_reason": None}]})
                await self._sse_write(response, None, {
                    **base, "choices": [{"index": 0, "delta": {},
                                         "finish_reason": choice.get("finish_reason") or "stop"}]})
                usage = final_resp.get("usage")
                if usage:
                    await self._sse_write(response, None, {**base, "choices": [], "usage": usage})
                await response.write(b"data: [DONE]\n\n")
        except (ConnectionResetError, ConnectionError):
            pass  # client disconnected
        finally:
            try:
                await response.write_eof()
            except Exception:
                pass
        return response

    @staticmethod
    async def _sse_write(response, event, payload):
        if event:
            line = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        else:
            line = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        await response.write(line.encode())

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
