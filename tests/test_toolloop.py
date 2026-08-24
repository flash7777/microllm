#!/usr/bin/env python3
"""Functional tests for the microllm builtin tool executor + PDF conversion.

Runs microllm as a subprocess (tests/config.yaml, port 18123) against four
in-process mocks:
  18192 fake vLLM backend (openai /v1/chat/completions + anthropic /v1/messages)
  18191 fake SearXNG (GET /search)
  18190 fake Taki (PUT /tika/pdf2chat, PUT /tika/text)
  18193 fake open_fetch (GET /fetch, text + kind="pdf" answers)

The fake vLLM is a state machine:
  round 1 -> uri_search tool call
  round 2 -> uri_fetch tool call (url https://example.com via fake open_fetch)
  round 3 -> final answer
  user text "ALWAYS-SEARCH" -> always a tool call (max-rounds test)
  user text "PDF-FETCH"     -> straight to uri_fetch of https://example.com/handbuch.pdf
                               (fake open_fetch answers kind="pdf" + base64)

All fetches run against the local fake open_fetch — no outbound network needed.

Usage:  python3 tests/test_toolloop.py
"""
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
from urllib.parse import urlparse

import aiohttp
from aiohttp import web

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MICROLLM_PY = os.path.join(REPO, "microllm.py")
WORK = "/tmp/microllm-test-work"
VLLM_LOG = os.path.join(WORK, "vllm_requests.jsonl")
MICROLLM_LOG = os.path.join(WORK, "microllm.log")
CHATLOG = os.path.join(WORK, "chatlog")
MICROLLM = "http://127.0.0.1:18123"

PNG_1X1 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
           "YAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
PDF_NORMAL = base64.b64encode(b"%PDF-1.4 NORMAL-TEST").decode()
PDF_FALLBACK = base64.b64encode(b"%PDF-1.4 FALLBACK-TEST").decode()

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"   [{str(detail)[:220]}]" if not cond and detail else ""))


# ---------- fake vLLM ----------

def _user_says(messages, marker):
    """True if any user message (str or content-block list) contains marker."""
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        if isinstance(content, str):
            if marker in content:
                return True
        elif isinstance(content, list):
            if any(marker in str(b.get("text", "")) for b in content if isinstance(b, dict)):
                return True
    return False


def _content_is_fetch_result(content):
    """Tool-result content: plain page text ('Content of ...') or a fetched
    PDF block list (text parts 'Fetched PDF ...' and/or image parts)."""
    if isinstance(content, str):
        return content.startswith("Content of")
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") in ("image", "image_url"):
                return True
            if b.get("type") == "text" and b.get("text", "").startswith("Fetched PDF"):
                return True
    return False


def openai_state(messages):
    always = any("ALWAYS-SEARCH" in str(m.get("content", ""))
                 for m in messages if m.get("role") == "user")
    if always:
        return "search"
    has_fetch = any(m.get("role") == "tool" and _content_is_fetch_result(m.get("content"))
                    for m in messages)
    if has_fetch:
        return "final"
    has_search = any(m.get("role") == "tool" and str(m.get("content", "")).startswith("Web search")
                     for m in messages)
    if _user_says(messages, "PDF-FETCH"):
        # PDF tests skip the search round and go straight to the PDF fetch
        return "fetch"
    if has_search:
        return "fetch"
    return "search"


def openai_tool(model, call_id, name, args):
    return {"id": f"chatcmpl-{call_id}", "object": "chat.completion", "created": 1, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                        "tool_calls": [{"id": call_id, "type": "function",
                                        "function": {"name": name, "arguments": json.dumps(args)}}]},
                        "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def openai_final(model):
    return {"id": "chatcmpl-final", "object": "chat.completion", "created": 1, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "FINAL-ANSWER-OK"},
                        "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


# ---------- OpenAI SSE stream builders (fake vLLM emits real SSE when stream=True) ----------

def openai_sse_events_tool(model, call_id, name, args):
    base = {"id": f"chatcmpl-{call_id}", "object": "chat.completion.chunk",
            "created": 1, "model": model}
    return [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": None},
                              "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": call_id, "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}]},
                              "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]


def openai_sse_events_final(model, text):
    base = {"id": "chatcmpl-final", "object": "chat.completion.chunk",
            "created": 1, "model": model}
    return [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": text},
                              "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]


def openai_sse_response(events):
    parts = [f"data: {json.dumps(ev)}\n\n" for ev in events]
    parts.append("data: [DONE]\n\n")
    return web.Response(text="".join(parts), content_type="text/event-stream")


def anthropic_state(messages):
    has_fetch, has_search = False, False
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    c = b.get("content", "")
                    if _content_is_fetch_result(c):
                        has_fetch = True
                    elif isinstance(c, str) and c.startswith("Web search"):
                        has_search = True
                    else:
                        has_search = True
    if has_fetch:
        return "final"
    if _user_says(messages, "PDF-FETCH") or has_search:
        return "fetch"
    return "search"


def anthropic_tool(model, tool_id, name, args):
    return {"id": f"msg-{tool_id}", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": args}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5}}


def anthropic_final(model):
    return {"id": "msg-final", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": "FINAL-ANSWER-OK"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}}


async def log_request(path, body):
    with open(VLLM_LOG, "a") as f:
        f.write(json.dumps({"path": path, "body": body}, ensure_ascii=False) + "\n")


def make_vllm_app():
    app = web.Application()

    async def models(_):
        return web.json_response({"data": [{"id": "fake-model"}]})

    async def chat(request):
        body = await request.json()
        await log_request("/v1/chat/completions", body)
        model = body.get("model", "fake-model")
        state = openai_state(body.get("messages", []))
        stream = bool(body.get("stream"))
        if state == "search":
            args = {"query": "OpenCloud KOSMOS"}
            if stream:
                return openai_sse_response(openai_sse_events_tool(model, "call_1", "uri_search", args))
            return web.json_response(openai_tool(model, "call_1", "uri_search", args))
        if state == "fetch":
            pdf_mode = _user_says(body.get("messages", []), "PDF-FETCH")
            args = {"url": "https://example.com/handbuch.pdf"} if pdf_mode \
                else {"url": "https://example.com"}
            if stream:
                return openai_sse_response(openai_sse_events_tool(model, "call_2", "uri_fetch", args))
            return web.json_response(openai_tool(model, "call_2", "uri_fetch", args))
        if stream:
            return openai_sse_response(openai_sse_events_final(model, "FINAL-ANSWER-OK"))
        return web.json_response(openai_final(model))

    async def messages(request):
        body = await request.json()
        await log_request("/v1/messages", body)
        model = body.get("model", "fake-model")
        state = anthropic_state(body.get("messages", []))
        if state == "search":
            return web.json_response(anthropic_tool(model, "toolu_1", "uri_search", {"query": "OpenCloud KOSMOS"}))
        if state == "fetch":
            pdf_mode = _user_says(body.get("messages", []), "PDF-FETCH")
            url = "https://example.com/handbuch.pdf" if pdf_mode else "https://example.com"
            return web.json_response(anthropic_tool(model, "toolu_2", "uri_fetch", {"url": url}))
        return web.json_response(anthropic_final(model))

    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/v1/messages", messages)
    return app


def make_searxng_app():
    app = web.Application()

    async def search(_):
        return web.json_response({"results": [
            {"title": "OpenCloud KOSMOS", "url": "https://kosmos.example.com", "content": "KOSMOS cloud platform."},
            {"title": "OpenCloud EU", "url": "https://opencloud.eu", "content": "European cloud."},
        ]})

    app.router.add_get("/search", search)
    return app


def make_taki_app():
    app = web.Application()

    async def pdf2chat(request):
        body = await request.read()
        if b"FALLBACK" in body:
            raise web.HTTPNotFound()
        return web.json_response({
            "text": "[Seite 1/2]\nHallo Welt\n[Seite 2/2]\nZweite Seite",
            "pages": 2,
            "images": [{"page": 1, "b64": PNG_1X1, "mime": "image/png"}],
        })

    async def text(request):
        await request.read()
        return web.json_response({"X-TIKA:content": "fallback text"})

    app.router.add_put("/tika/pdf2chat", pdf2chat)
    app.router.add_put("/tika/text", text)
    return app


def make_webfetch_app():
    """Fake open_fetch service (bot-safe fetcher): answers text pages, and
    kind="pdf" + base64 for the handbuch.pdf test URL (like the real
    service since the PDF feature). Private/loopback URLs are refused."""
    app = web.Application()
    pdf_bytes = base64.b64decode(PDF_NORMAL)

    async def fetch(request):
        url = request.query.get("url", "")
        host = urlparse(url).hostname or ""
        if host in ("127.0.0.1", "localhost") or host.startswith("192.168."):
            return web.json_response({"ok": False,
                                      "error": f"URL not allowed (only public http/https): {url}"})
        if url.endswith("/handbuch.pdf"):
            return web.json_response({
                "ok": True, "kind": "pdf", "status": 200, "final_url": url,
                "content_type": "application/pdf", "bytes": len(pdf_bytes),
                "b64": PDF_NORMAL, "text": f"[PDF-Datei: {len(pdf_bytes)} bytes]"})
        return web.json_response({
            "ok": True, "status": 200, "final_url": url,
            "content_type": "text/html", "bytes": 24,
            "text": "Example page text (fake open_fetch)"})

    async def health(_):
        return web.json_response({"status": "ok"})

    app.router.add_get("/fetch", fetch)
    app.router.add_get("/health", health)
    return app


# ---------- helpers ----------

def read_vllm_log():
    if not os.path.exists(VLLM_LOG):
        return []
    with open(VLLM_LOG) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def find_marker(marker, path=None):
    return [e for e in read_vllm_log()
            if (path is None or e["path"] == path) and marker in json.dumps(e["body"], ensure_ascii=False)]


async def post(sess, path, payload):
    async with sess.post(MICROLLM + path, json=payload) as resp:
        return resp.status, resp.headers.get("content-type", ""), await resp.text()


# ---------- tests ----------

async def t1_openai_tool_loop(sess):
    print("T1: openai non-stream tool loop (search -> fetch -> answer)")
    marker = "MK-T1"
    status, _ct, text = await post(sess, "/v1/chat/completions", {
        "model": "testgroup", "stream": False,
        "messages": [{"role": "user", "content": f"Was ist OpenCloud? {marker}"}]})
    body = json.loads(text)
    content = body.get("choices", [{}])[0].get("message", {}).get("content")
    check("T1 status 200", status == 200, text[:200])
    check("T1 final answer", content == "FINAL-ANSWER-OK", str(content)[:200])
    entries = find_marker(marker, "/v1/chat/completions")
    check("T1 3 backend rounds", len(entries) == 3, len(entries))
    if len(entries) == 3:
        r1, r2, r3 = (e["body"] for e in entries)
        t1 = [(t.get("function") or {}).get("name") for t in r1.get("tools", []) if isinstance(t, dict)]
        check("T1 r1 builtin tools injected", t1 == ["uri_search", "uri_fetch"], t1)
        check("T1 r1 stream forced off", r1.get("stream") is False, r1.get("stream"))
        check("T1 r1 model rewritten", r1.get("model") == "fake-model", r1.get("model"))
        m2 = r2.get("messages", [])
        check("T1 r2 carries tool result", any(m.get("role") == "tool" for m in m2))
        check("T1 r2 tools still present", bool(r2.get("tools")))
        check("T1 r3 tools dropped", "tools" not in r3, list(r3.keys()))
        check("T1 r3 has fetch result",
              any(str(m.get("content", "")).startswith("Content of")
                  for m in r3.get("messages", []) if m.get("role") == "tool"))
    files = os.listdir(CHATLOG) if os.path.isdir(CHATLOG) else []
    check("T1 chatlog round files",
          any("req_r1" in f for f in files) and any("req_r3" in f for f in files), files[:10])


async def t2_openai_stream(sess):
    print("T2: openai stream -> synthesized SSE")
    marker = "MK-T2"
    async with sess.post(MICROLLM + "/v1/chat/completions", json={
        "model": "testgroup", "stream": True,
        "messages": [{"role": "user", "content": f"Hallo {marker}"}]}) as resp:
        ct = resp.headers.get("content-type", "")
        raw = await resp.text()
    check("T2 sse content-type", "text/event-stream" in ct, ct)
    check("T2 done marker", "data: [DONE]" in raw, raw[-120:])
    chunks = []
    for ln in raw.splitlines():
        if ln.startswith("data: ") and ln[6:] != "[DONE]":
            try:
                chunks.append(json.loads(ln[6:]))
            except json.JSONDecodeError:
                pass
    parts, finish = [], None
    for c in chunks:
        for ch in c.get("choices", []):
            d = ch.get("delta", {})
            if d.get("content"):
                parts.append(d["content"])
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    content = "".join(parts)
    check("T2 streamed final answer", "FINAL-ANSWER-OK" in content, content[:200])
    check("T2 tool trace details", '<details type="tool_calls" done="true"' in content, content[:300])
    check("T2 tool trace search", 'name="uri_search"' in content, content[:300])
    check("T2 tool trace fetch", 'name="uri_fetch"' in content, content[:300])
    check("T2 finish stop", finish == "stop", finish)
    check("T2 3 backend rounds", len(find_marker(marker, "/v1/chat/completions")) == 3)


async def t3_anthropic_tool_loop(sess):
    print("T3: anthropic /v1/messages tool loop")
    marker = "MK-T3"
    status, _ct, text = await post(sess, "/v1/messages", {
        "model": "testgroup",
        "messages": [{"role": "user", "content": f"Was ist OpenCloud? {marker}"}]})
    body = json.loads(text)
    texts = [b.get("text") for b in body.get("content", [])
             if isinstance(b, dict) and b.get("type") == "text"]
    check("T3 status 200", status == 200, text[:200])
    check("T3 final answer", texts == ["FINAL-ANSWER-OK"], texts[:1] if texts else text[:200])
    entries = find_marker(marker, "/v1/messages")
    check("T3 3 backend rounds", len(entries) == 3, len(entries))
    if len(entries) == 3:
        r1, r2, r3 = (e["body"] for e in entries)
        check("T3 r1 anthropic tools",
              [t.get("name") for t in r1.get("tools", [])] == ["uri_search", "uri_fetch"],
              [t.get("name") for t in r1.get("tools", [])])
        check("T3 r1 input_schema shape", all("input_schema" in t for t in r1.get("tools", [])))
        m2 = r2.get("messages", [])
        check("T3 r2 assistant tool_use",
              any(m.get("role") == "assistant" and isinstance(m.get("content"), list)
                  and any(isinstance(b, dict) and b.get("type") == "tool_use" for b in m["content"])
                  for m in m2))
        check("T3 r2 user tool_result",
              any(m.get("role") == "user" and isinstance(m.get("content"), list)
                  and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])
                  for m in m2))
        check("T3 r3 tools dropped", "tools" not in r3, list(r3.keys()))


async def t4_plain_passthrough(sess):
    print("T4: group without builtin -> plain passthrough (no execution)")
    marker = "MK-T4"
    status, _ct, text = await post(sess, "/v1/chat/completions", {
        "model": "plain", "messages": [{"role": "user", "content": f"Hallo {marker}"}]})
    body = json.loads(text)
    tc = body.get("choices", [{}])[0].get("message", {}).get("tool_calls")
    check("T4 status 200", status == 200, text[:200])
    check("T4 raw tool_calls passed through",
          bool(tc) and tc[0].get("function", {}).get("name") == "uri_search", text[:200])
    entries = find_marker(marker, "/v1/chat/completions")
    check("T4 single backend call", len(entries) == 1, len(entries))
    check("T4 no tools injected", "tools" not in entries[0]["body"] if entries else False)


async def t5_pdf_openai_input_file(sess):
    print("T5: openai input_file PDF -> text + image blocks")
    marker = "MK-T5"
    status, _ct, text = await post(sess, "/v1/chat/completions", {
        "model": "testgroup", "stream": False,
        "messages": [{"role": "user", "content": [
            {"type": "input_file",
             "file": {"filename": "handbuch.pdf", "file_data": f"data:application/pdf;base64,{PDF_NORMAL}"}},
            {"type": "text", "text": f"Fasse das Handbuch zusammen {marker}"},
        ]}]})
    check("T5 status 200", status == 200, text[:200])
    entries = find_marker(marker, "/v1/chat/completions")
    if not entries:
        check("T5 backend saw request", False)
        return
    r1 = entries[0]["body"]
    uc = r1["messages"][0]["content"]
    types = [b.get("type") for b in uc]
    check("T5 block order text,image,text", types == ["text", "image_url", "text"], types)
    check("T5 page markers", "[Seite 1/2]" in uc[0]["text"] and "[Seite 2/2]" in uc[0]["text"],
          uc[0]["text"][:120])
    check("T5 filename label", uc[0]["text"].startswith("[PDF handbuch.pdf]"), uc[0]["text"][:60])
    check("T5 image data url",
          uc[1].get("image_url", {}).get("url", "").startswith("data:image/png;base64,")
          and PNG_1X1 in uc[1].get("image_url", {}).get("url", ""))
    check("T5 input_file gone", all(b.get("type") != "input_file" for b in uc), types)
    check("T5 loop finished (3 rounds)", len(entries) == 3, len(entries))


async def t6_pdf_anthropic_document(sess):
    print("T6: anthropic document block -> text + image blocks")
    marker = "MK-T6"
    status, _ct, text = await post(sess, "/v1/messages", {
        "model": "testgroup",
        "messages": [{"role": "user", "content": [
            {"type": "document",
             "source": {"type": "base64", "media_type": "application/pdf", "data": PDF_NORMAL}},
            {"type": "text", "text": f"Zusammenfassung {marker}"},
        ]}]})
    check("T6 status 200", status == 200, text[:200])
    entries = find_marker(marker, "/v1/messages")
    if not entries:
        check("T6 backend saw request", False)
        return
    uc = entries[0]["body"]["messages"][0]["content"]
    types = [b.get("type") for b in uc]
    check("T6 block order text,image,text", types == ["text", "image", "text"], types)
    check("T6 page markers", "[Seite 1/2]" in uc[0]["text"], uc[0]["text"][:120])
    check("T6 anthropic image block",
          uc[1].get("source", {}).get("type") == "base64"
          and uc[1].get("source", {}).get("data") == PNG_1X1)


async def t7_pdf_fallback_tika_text(sess):
    print("T7: pdf2chat 404 -> fallback /tika/text (text only)")
    marker = "MK-T7"
    status, _ct, text = await post(sess, "/v1/chat/completions", {
        "model": "testgroup", "stream": False,
        "messages": [{"role": "user", "content": [
            {"type": "input_file",
             "file": {"filename": "alt.pdf", "file_data": f"data:application/pdf;base64,{PDF_FALLBACK}"}},
            {"type": "text", "text": f"Lesen {marker}"},
        ]}]})
    check("T7 status 200", status == 200, text[:200])
    entries = find_marker(marker, "/v1/chat/completions")
    if not entries:
        check("T7 backend saw request", False)
        return
    uc = entries[0]["body"]["messages"][0]["content"]
    types = [b.get("type") for b in uc]
    check("T7 text + user text only", types == ["text", "text"], types)
    check("T7 fallback text", "fallback text" in uc[0]["text"], uc[0]["text"][:160])
    check("T7 no image", "image_url" not in types, types)


async def t8_alias_inheritance(sess):
    print("T8: alias group inherits builtin + pdf meta")
    marker = "MK-T8"
    status, _ct, text = await post(sess, "/v1/chat/completions", {
        "model": "testalias", "stream": False,
        "messages": [{"role": "user", "content": f"Hallo {marker}"}]})
    body = json.loads(text)
    content = body.get("choices", [{}])[0].get("message", {}).get("content")
    check("T8 status 200", status == 200, text[:200])
    check("T8 final answer", content == "FINAL-ANSWER-OK", str(content)[:200])
    check("T8 3 backend rounds", len(find_marker(marker, "/v1/chat/completions")) == 3)


async def t9_units():
    print("T9: unit tests (SSRF guard, html2text, web_fetch loopback)")
    sys.path.insert(0, REPO)
    import microllm as m
    safe = m.MicroLLM._url_is_safe
    check("T9 loopback blocked", safe("http://127.0.0.1:8012") is False)
    check("T9 private 192.168 blocked", safe("http://192.168.1.10/") is False)
    check("T9 private 10.x blocked", safe("http://10.30.10.111/") is False)
    check("T9 link-local 169.254 blocked", safe("http://169.254.169.254/latest") is False)
    check("T9 public allowed", safe("https://example.com") is True)
    check("T9 ftp blocked", safe("ftp://example.com") is False)
    check("T9 no host blocked", safe("http://") is False)
    html = ('<html><head><title>X</title></head><body><script>var a=1;</script>'
            '<h1>Title</h1><p>Para <a href="https://x.com/y">link</a></p>'
            '<div>D1</div><div>D2</div></body></html>')
    text = m._html_to_text(html)
    check("T9 script stripped", "var a=1" not in text, text)
    check("T9 title skipped", "X" not in text.replace("Title", ""), text)
    check("T9 link with url", "link (https://x.com/y)" in text, text)
    check("T9 blocks separated", "D1" in text and "D2" in text and text.index("D1") < text.index("D2"), text)

    proxy = m.MicroLLM.__new__(m.MicroLLM)
    proxy.session = aiohttp.ClientSession()
    proxy.web_fetch_url = None
    try:
        res = await proxy._web_fetch("http://127.0.0.1:18191/search?q=x")
        check("T9 web_fetch loopback refused", res.startswith("Error: URL not allowed"), res[:120])
    finally:
        await proxy.session.close()


async def t10_max_rounds_forced(sess):
    print("T10: model stuck in tool calls -> forced final after max rounds")
    marker = "MK-T10"
    status, _ct, text = await post(sess, "/v1/chat/completions", {
        "model": "testgroup", "stream": False,
        "messages": [{"role": "user", "content": f"ALWAYS-SEARCH {marker}"}]})
    body = json.loads(text)
    tc = body.get("choices", [{}])[0].get("message", {}).get("tool_calls")
    check("T10 status 200", status == 200, text[:200])
    check("T10 loop terminated (raw last round back)", bool(tc), text[:200])
    check("T10 exactly 3 rounds", len(find_marker(marker, "/v1/chat/completions")) == 3,
          len(find_marker(marker, "/v1/chat/completions")))


async def t11_pdf_fetch_openai(sess):
    print("T11: uri_fetch of a PDF (openai) -> multimodal tool result")
    marker = "MK-T11"
    status, _ct, text = await post(sess, "/v1/chat/completions", {
        "model": "testgroup", "stream": False,
        "messages": [{"role": "user", "content": f"PDF-FETCH {marker}"}]})
    body = json.loads(text)
    content = body.get("choices", [{}])[0].get("message", {}).get("content")
    check("T11 status 200", status == 200, text[:200])
    check("T11 final answer", content == "FINAL-ANSWER-OK", str(content)[:200])
    entries = find_marker(marker, "/v1/chat/completions")
    check("T11 2 backend rounds (search skipped)", len(entries) == 2, len(entries))
    if len(entries) == 2:
        r2 = entries[1]["body"]
        tool_msgs = [m for m in r2.get("messages", []) if m.get("role") == "tool"]
        check("T11 one tool result", len(tool_msgs) == 1, len(tool_msgs))
        c = tool_msgs[0].get("content")
        check("T11 tool content is block list", isinstance(c, list), type(c))
        if isinstance(c, list):
            types = [b.get("type") for b in c if isinstance(b, dict)]
            check("T11 blocks text,text,image", types == ["text", "text", "image_url"], types)
            check("T11 url header", c[0]["text"].startswith("Fetched PDF https://example.com/handbuch.pdf"),
                  c[0]["text"][:100])
            check("T11 tika text", "[Seite 1/2]" in c[1]["text"] and "Hallo Welt" in c[1]["text"],
                  c[1]["text"][:120])
            check("T11 filename label", "[PDF handbuch.pdf]" in c[1]["text"], c[1]["text"][:60])
            check("T11 image data url",
                  c[2].get("image_url", {}).get("url", "").startswith("data:image/png;base64,")
                  and PNG_1X1 in c[2].get("image_url", {}).get("url", ""))


async def t12_pdf_fetch_stream(sess):
    print("T12: uri_fetch of a PDF (stream) -> trace without base64")
    marker = "MK-T12"
    async with sess.post(MICROLLM + "/v1/chat/completions", json={
        "model": "testgroup", "stream": True,
        "messages": [{"role": "user", "content": f"PDF-FETCH {marker}"}]}) as resp:
        ct = resp.headers.get("content-type", "")
        raw = await resp.text()
    check("T12 sse content-type", "text/event-stream" in ct, ct)
    check("T12 done marker", "data: [DONE]" in raw, raw[-120:])
    chunks = []
    for ln in raw.splitlines():
        if ln.startswith("data: ") and ln[6:] != "[DONE]":
            try:
                chunks.append(json.loads(ln[6:]))
            except json.JSONDecodeError:
                pass
    parts = []
    for c in chunks:
        for ch in c.get("choices", []):
            if ch.get("delta", {}).get("content"):
                parts.append(ch["delta"]["content"])
    content = "".join(parts)
    check("T12 streamed final answer", "FINAL-ANSWER-OK" in content, content[:200])
    check("T12 trace fetch card", 'name="uri_fetch"' in content, content[:300])
    check("T12 trace tika text", "Hallo Welt" in content, content[:400])
    check("T12 trace image counter", "1 images passed to the model" in content, content[:400])
    check("T12 no base64 in stream", PNG_1X1 not in content and "iVBORw0KGgo" not in content,
          content[:400])
    check("T12 2 backend rounds", len(find_marker(marker, "/v1/chat/completions")) == 2,
          len(find_marker(marker, "/v1/chat/completions")))


async def t13_pdf_fetch_anthropic(sess):
    print("T13: uri_fetch of a PDF (anthropic) -> tool_result block list")
    marker = "MK-T13"
    status, _ct, text = await post(sess, "/v1/messages", {
        "model": "testgroup",
        "messages": [{"role": "user", "content": f"PDF-FETCH {marker}"}]})
    body = json.loads(text)
    texts = [b.get("text") for b in body.get("content", [])
             if isinstance(b, dict) and b.get("type") == "text"]
    check("T13 status 200", status == 200, text[:200])
    check("T13 final answer", texts == ["FINAL-ANSWER-OK"], texts[:1] if texts else text[:200])
    entries = find_marker(marker, "/v1/messages")
    check("T13 2 backend rounds", len(entries) == 2, len(entries))
    if len(entries) == 2:
        m2 = entries[1]["body"].get("messages", [])
        tool_results = [b for m in m2 if m.get("role") == "user" and isinstance(m.get("content"), list)
                        for b in m["content"]
                        if isinstance(b, dict) and b.get("type") == "tool_result"]
        check("T13 one tool_result", len(tool_results) == 1, len(tool_results))
        c = tool_results[0].get("content") if tool_results else None
        check("T13 tool_result is block list", isinstance(c, list), type(c))
        if isinstance(c, list):
            types = [b.get("type") for b in c if isinstance(b, dict)]
            check("T13 blocks text,text,image", types == ["text", "text", "image"], types)
            check("T13 anthropic image block",
                  c[2].get("source", {}).get("type") == "base64"
                  and c[2].get("source", {}).get("data") == PNG_1X1)


async def t14_pdf_units():
    print("T14: unit tests (tool result text, fetched PDF blocks, in-process PDF fetch)")
    sys.path.insert(0, REPO)
    import microllm as m

    blocks = [{"type": "text", "text": "abc"},
              {"type": "image_url", "image_url": {"url": "data:image/png;base64,QQ=="}},
              {"type": "text", "text": "def"}]
    txt = m.MicroLLM._tool_result_text(blocks)
    check("T14 result text from parts", txt.startswith("abc\ndef"), txt)
    check("T14 image counter", "[1 images passed to the model]" in txt, txt)
    check("T14 no b64 in trace text", "QQ==" not in txt, txt)
    check("T14 str passthrough", m.MicroLLM._tool_result_text("plain") == "plain")
    check("T14 truncation", len(m.MicroLLM._tool_result_text("x" * 5000)) < 2100,
          len(m.MicroLLM._tool_result_text("x" * 5000)))

    # in-process _web_fetch: PDF content-type -> b64 dict (SSRF patched, local server)
    pdf_bytes = base64.b64decode(PDF_NORMAL)

    async def serve_pdf(_):
        return web.Response(body=pdf_bytes, content_type="application/pdf")

    _app = web.Application()
    _app.router.add_get("/x.pdf", serve_pdf)
    runner = web.AppRunner(_app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 18194)
    await site.start()
    proxy = m.MicroLLM.__new__(m.MicroLLM)
    proxy.session = aiohttp.ClientSession()
    proxy.web_fetch_url = None
    proxy.ocr_url = "http://127.0.0.1:18190"
    proxy._url_is_safe = lambda u: True
    try:
        res = await proxy._web_fetch("http://127.0.0.1:18194/x.pdf")
        check("T14 in-process pdf is dict",
              isinstance(res, dict) and res.get("kind") == "pdf", str(res)[:120])
        check("T14 in-process pdf b64",
              isinstance(res, dict) and base64.b64decode(res.get("b64", "")) == pdf_bytes)

        # _fetched_pdf_to_blocks: empty b64 -> error, b64 -> pipeline blocks (fake Taki 18190)
        err = await proxy._fetched_pdf_to_blocks(
            {"kind": "pdf", "url": "https://x.com/a.pdf", "bytes": 99, "b64": ""},
            "https://x.com/a.pdf", None, False)
        check("T14 empty b64 -> error", isinstance(err, str) and err.startswith("Error: PDF too large"),
              str(err)[:120])
        blocks2 = await proxy._fetched_pdf_to_blocks(
            {"kind": "pdf", "url": "https://x.com/handbuch.pdf", "bytes": len(pdf_bytes), "b64": PDF_NORMAL},
            "https://x.com/handbuch.pdf",
            {"pdf": {"enabled": True, "images": True, "vision": True, "dpi": 100, "max_image_pages": 8}},
            False)
        check("T14 fetched pdf blocks", isinstance(blocks2, list) and len(blocks2) == 3,
              str(blocks2)[:120] if not isinstance(blocks2, list) else [b.get("type") for b in blocks2])
        if isinstance(blocks2, list) and len(blocks2) == 3:
            check("T14 block order", [b.get("type") for b in blocks2] == ["text", "text", "image_url"])
            check("T14 url header", blocks2[0]["text"].startswith("Fetched PDF https://x.com/handbuch.pdf"),
                  blocks2[0]["text"][:100])
            check("T14 tika text in block", "Hallo Welt" in blocks2[1]["text"], blocks2[1]["text"][:120])
    finally:
        await proxy.session.close()
        await runner.cleanup()


# ---------- main ----------

async def main():
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(CHATLOG, exist_ok=True)

    runners = []
    for app, port in ((make_vllm_app(), 18192), (make_searxng_app(), 18191),
                      (make_taki_app(), 18190), (make_webfetch_app(), 18193)):
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()
        runners.append(runner)
    print("mocks up (vllm 18192, searxng 18191, taki 18190, webfetch 18193)")

    logf = open(MICROLLM_LOG, "w")
    proc = subprocess.Popen([sys.executable, MICROLLM_PY, os.path.join(HERE, "config.yaml"), "18123"],
                            stdout=logf, stderr=subprocess.STDOUT)
    try:
        healthy = False
        for _ in range(50):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(MICROLLM + "/health",
                                     timeout=aiohttp.ClientTimeout(total=1)) as r:
                        if r.status == 200:
                            healthy = True
                            break
            except Exception:
                pass
            await asyncio.sleep(0.2)
        if not healthy:
            print("FATAL: microllm not healthy")
            print(open(MICROLLM_LOG).read()[-2000:])
            return 1

        async with aiohttp.ClientSession() as sess:
            await t1_openai_tool_loop(sess)
            await t2_openai_stream(sess)
            await t3_anthropic_tool_loop(sess)
            await t4_plain_passthrough(sess)
            await t5_pdf_openai_input_file(sess)
            await t6_pdf_anthropic_document(sess)
            await t7_pdf_fallback_tika_text(sess)
            await t8_alias_inheritance(sess)
            await t9_units()
            await t10_max_rounds_forced(sess)
            await t11_pdf_fetch_openai(sess)
            await t12_pdf_fetch_stream(sess)
            await t13_pdf_fetch_anthropic(sess)
            await t14_pdf_units()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        logf.close()
        for runner in runners:
            await runner.cleanup()

    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED:")
        for n in failed:
            print(f"  - {n}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
