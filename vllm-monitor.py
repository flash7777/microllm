#!/usr/bin/env python3
"""vLLM MicroLLM Monitor — TUI dashboard.

Aggregates data from three sources:
  1. MicroLLM /stats  — proxy-level routing stats, per-model request counts
  2. vLLM /metrics     — Prometheus metrics per backend (KV cache, tok/s, latency)
  3. Podman            — container status for all vLLM backends

Usage:
  python3 vllm-monitor.py                          # auto-detect from microllm config
  python3 vllm-monitor.py --microllm http://localhost:8012
  python3 vllm-monitor.py --backends http://localhost:8011
"""

import argparse
import json
import re
import subprocess
import time
import urllib.request
from collections import defaultdict
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box


# ── Data fetchers ──────────────────────────────────────────────────────────

def fetch_json(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read()), None
    except Exception as e:
        return None, str(e)


def fetch_prometheus(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            text = r.read().decode()
    except Exception as e:
        return None, str(e)
    metrics = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r'^([\w:]+)(?:\{([^}]*)\})?\s+([\d.eE+\-]+)$', line)
        if m:
            name = m.group(1)
            labels = m.group(2) or ""
            raw = m.group(3)
            val = float(raw) if '.' in raw or 'e' in raw.lower() else int(raw)
            metrics.setdefault(name, []).append((labels, val))
    return metrics, None


def fetch_podman_containers():
    """Get status of all vllm-related podman containers."""
    try:
        result = subprocess.run(
            ["podman", "ps", "--format",
             "{{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}"],
            capture_output=True, text=True, timeout=5,
        )
        containers = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) >= 4:
                name, status, ports, image = parts[0], parts[1], parts[2], parts[3]
                if "vllm" in name.lower() or "vllm" in image.lower():
                    containers.append({
                        "name": name,
                        "status": status,
                        "ports": ports,
                        "image": image.split("/")[-1][:30],
                    })
        return containers
    except Exception:
        return []


def fetch_nvidia_gpu():
    """Get GPU utilization and memory from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "util": int(parts[0]),
                    "mem_used": int(float(parts[1])),
                    "mem_total": int(float(parts[2])),
                    "temp": int(float(parts[3])),
                    "power": float(parts[4]),
                })
        return gpus
    except Exception:
        return []


def parse_microllm_config(path):
    """Extract backend URLs from microllm config.yaml."""
    try:
        import yaml
        with open(path) as f:
            config = yaml.safe_load(f)
        backends = set()
        for entry in config.get("model_list", []):
            params = entry.get("litellm_params", entry.get("params", {}))
            api_base = params.get("api_base", "").rstrip("/")
            if api_base.endswith("/v1"):
                api_base = api_base[:-3]
            if api_base and "localhost" in api_base:
                backends.add(api_base)
        port = config.get("general_settings", {}).get("port", 8012)
        return list(backends), port
    except Exception:
        return [], 8012


# ── Helpers ────────────────────────────────────────────────────────────────

def mv(metrics, name, idx=0):
    entries = metrics.get(name, [])
    return entries[idx][1] if idx < len(entries) else 0


def mv_label(metrics, name, key, value):
    for labels, val in metrics.get(name, []):
        if f'{key}="{value}"' in labels:
            return val
    return 0


def bar(pct, width=25):
    filled = int(pct / 100 * width)
    color = "green" if pct < 60 else "yellow" if pct < 85 else "red"
    return f"[{color}]{'█' * filled}{'░' * (width - filled)}[/] {pct:.1f}%"


def fmt_tok(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(int(n))


# ── Dashboard builder ──────────────────────────────────────────────────────

def build_dashboard(state):
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="top", size=7),
        Layout(name="mid"),
        Layout(name="footer", size=3),
    )
    layout["mid"].split_row(
        Layout(name="mid_left", ratio=3),
        Layout(name="mid_right", ratio=2),
    )

    now_str = datetime.now().strftime("%H:%M:%S")

    # ── Header ──
    parts = [f" MicroLLM Monitor  |  {now_str}"]
    if state["microllm_stats"]:
        ms = state["microllm_stats"]
        parts.append(f"uptime {ms['uptime_s']//3600}h{(ms['uptime_s']%3600)//60}m")
        parts.append(f"{ms['routes']} routes")
    parts.append("[green]LIVE[/]" if any(state["backend_metrics"].values()) else "[yellow]POLLING[/]")
    layout["header"].update(Panel(
        Text.from_markup("  |  ".join(parts)),
        style="bold blue",
    ))

    # ── Top: GPU + Containers ──
    top_table = Table(box=box.SIMPLE_HEAVY, expand=True, show_edge=False)
    top_table.add_column("GPU", style="bold cyan", width=8)
    top_table.add_column("Util", width=32)
    top_table.add_column("VRAM", width=32)
    top_table.add_column("Temp", justify="right", width=6)
    top_table.add_column("Power", justify="right", width=8)
    top_table.add_column("  Container", style="bold", width=28)
    top_table.add_column("Status", width=20)

    gpus = state["gpus"]
    containers = state["containers"]
    rows = max(len(gpus), len(containers), 1)

    for i in range(rows):
        gpu_cols = ("", "", "", "", "")
        if i < len(gpus):
            g = gpus[i]
            mem_pct = g["mem_used"] / g["mem_total"] * 100 if g["mem_total"] else 0
            gpu_cols = (
                f"GPU {i}",
                bar(g["util"]),
                bar(mem_pct) + f" {g['mem_used']//1024}/{g['mem_total']//1024}G",
                f"{g['temp']}°C",
                f"{g['power']:.0f}W",
            )
        ct_cols = ("", "")
        if i < len(containers):
            c = containers[i]
            up = "[green]Up[/]" if c["status"].startswith("Up") else "[red]Down[/]"
            ct_cols = (f"  {c['name']}", f"{up}  {c['ports'][:25]}")
        top_table.add_row(*gpu_cols, *ct_cols)

    layout["top"].update(Panel(top_table, title="Infrastructure", border_style="cyan"))

    # ── Mid Left: Backend vLLM details ──
    for url, bm in state["backend_metrics"].items():
        if bm is None:
            continue

        running = mv(bm, "vllm:num_requests_running")
        waiting = mv(bm, "vllm:num_requests_waiting")
        kv_pct = mv(bm, "vllm:kv_cache_usage_perc") * 100
        prompt_total = mv(bm, "vllm:prompt_tokens_total")
        gen_total = mv(bm, "vllm:generation_tokens_total")
        cache_hits = mv(bm, "vllm:prefix_cache_hits_total")
        cache_queries = mv(bm, "vllm:prefix_cache_queries_total")
        cache_pct = (cache_hits / cache_queries * 100) if cache_queries > 0 else 0
        req_count = mv(bm, "vllm:e2e_request_latency_seconds_count")
        req_sum = mv(bm, "vllm:e2e_request_latency_seconds_sum")
        avg_lat = (req_sum / req_count) if req_count > 0 else 0
        ttft_n = mv(bm, "vllm:time_to_first_token_seconds_count")
        ttft_s = mv(bm, "vllm:time_to_first_token_seconds_sum")
        avg_ttft = (ttft_s / ttft_n) if ttft_n > 0 else 0
        itl_n = mv(bm, "vllm:inter_token_latency_seconds_count")
        itl_s = mv(bm, "vllm:inter_token_latency_seconds_sum")
        avg_itl = (itl_s / itl_n) if itl_n > 0 else 0
        preemptions = mv(bm, "vllm:num_preemptions_total")
        success_stop = mv_label(bm, "vllm:request_success_total", "finished_reason", "stop")
        success_len = mv_label(bm, "vllm:request_success_total", "finished_reason", "length")
        success_abort = mv_label(bm, "vllm:request_success_total", "finished_reason", "abort")

        # Rates
        prev = state["prev_backends"].get(url, {})
        dt = state["dt"]
        prompt_rate = (prompt_total - prev.get("prompt", 0)) / dt if dt > 0 and prev else 0
        gen_rate = (gen_total - prev.get("gen", 0)) / dt if dt > 0 and prev else 0

        # Model name from metric labels
        model_name = ""
        for entries in bm.values():
            for labels, _ in entries:
                m = re.search(r'model_name="([^"]*)"', labels)
                if m:
                    model_name = m.group(1)
                    break
            if model_name:
                break

        bt = Table(show_header=False, box=None, padding=(0, 1))
        bt.add_column("k", style="bold", width=20)
        bt.add_column("v", width=38)
        bt.add_column("k2", style="bold", width=20)
        bt.add_column("v2", width=30)

        run_style = "bold green" if running > 0 else "dim"
        bt.add_row(
            "Requests", f"[{run_style}]{int(running)} running[/], {int(waiting)} waiting",
            "Prompt tok/s", f"[bold green]{prompt_rate:.1f}[/]" if prompt_rate > 0 else "[dim]idle[/]",
        )
        bt.add_row(
            "KV Cache", bar(kv_pct),
            "Gen tok/s", f"[bold green]{gen_rate:.1f}[/]" if gen_rate > 0 else "[dim]idle[/]",
        )
        bt.add_row(
            "Prefix Cache", f"{cache_pct:.0f}% hit ({fmt_tok(cache_hits)}/{fmt_tok(cache_queries)})",
            "Prompt Total", fmt_tok(prompt_total),
        )
        bt.add_row(
            "Avg Latency", f"{avg_lat:.2f}s" if avg_lat > 0 else "—",
            "Gen Total", fmt_tok(gen_total),
        )
        bt.add_row(
            "Avg TTFT", f"{avg_ttft:.3f}s" if avg_ttft > 0 else "—",
            "Finished", f"stop:{int(success_stop)} len:{int(success_len)}" + (f" [red]abort:{int(success_abort)}[/]" if success_abort else ""),
        )
        bt.add_row(
            "Avg ITL", f"{avg_itl*1000:.1f}ms" if avg_itl > 0 else "—",
            "Preemptions", f"[red]{int(preemptions)}[/]" if preemptions > 0 else "0",
        )

        port = url.split(":")[-1] if ":" in url else "?"
        layout["mid_left"].update(Panel(
            bt,
            title=f"vLLM Backend — {model_name} — :{port}",
            border_style="magenta",
        ))
        break  # For now show first backend; could stack multiple

    if not any(state["backend_metrics"].values()):
        layout["mid_left"].update(Panel("[dim]No vLLM backends reachable[/]", title="vLLM Backends", border_style="red"))

    # ── Mid Right: MicroLLM routes ──
    ms = state["microllm_stats"]
    if ms:
        rt = Table(box=box.SIMPLE, expand=True, show_edge=False)
        rt.add_column("Model", style="bold", width=22)
        rt.add_column("Reqs", justify="right", width=6)
        rt.add_column("Err", justify="right", width=5)
        rt.add_column("Tok In", justify="right", width=8)
        rt.add_column("Tok Out", justify="right", width=8)
        rt.add_column("tok/s", justify="right", width=7)
        rt.add_column("Avg Gen", justify="right", width=8)

        # Sort: active models first, then by request count
        models = sorted(
            ms.get("models", {}).items(),
            key=lambda x: (-x[1]["requests"], x[0]),
        )

        for name, s in models:
            if s["requests"] == 0:
                continue
            err_str = f"[red]{s['errors']}[/]" if s["errors"] > 0 else "[dim]0[/]"
            avg_gen = f"{s['total_gen_s']/s['requests']:.1f}s" if s["requests"] > 0 else "—"
            tok_s_str = f"[green]{s['last_tok_s']}[/]" if s["last_tok_s"] > 0 else f"[dim]{s['avg_tok_s']}[/]"
            rt.add_row(
                name,
                str(s["requests"]),
                err_str,
                fmt_tok(s["tokens_in"]),
                fmt_tok(s["tokens_out"]),
                tok_s_str,
                avg_gen,
            )

        # Inactive models summary
        inactive = sum(1 for _, s in models if s["requests"] == 0)
        if inactive:
            rt.add_row(f"[dim]+ {inactive} inactive routes[/]", "", "", "", "", "", "")

        layout["mid_right"].update(Panel(rt, title="MicroLLM Routes", border_style="green"))
    else:
        layout["mid_right"].update(Panel(
            "[dim]MicroLLM not reachable[/]",
            title="MicroLLM Routes",
            border_style="red",
        ))

    # ── Footer ──
    total_gen_rate = sum(
        ((mv(bm, "vllm:generation_tokens_total") - state["prev_backends"].get(u, {}).get("gen", 0)) / state["dt"]
         if state["dt"] > 0 and state["prev_backends"].get(u) and bm else 0)
        for u, bm in state["backend_metrics"].items()
    )
    layout["footer"].update(Panel(Text.from_markup(
        f"[dim]  Refresh: {state['interval']:.0f}s  |  Ctrl+C quit  |  "
        f"gen tok/s: [bold]{total_gen_rate:.1f}[/]  |  "
        f"backends: {sum(1 for v in state['backend_metrics'].values() if v)}/{len(state['backend_metrics'])}[/]"
    ), style="dim"))

    return layout


# ── Main loop ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MicroLLM + vLLM Monitor")
    parser.add_argument("--microllm", default=None, help="MicroLLM URL (default: auto-detect or http://localhost:8012)")
    parser.add_argument("--backends", nargs="*", default=None, help="vLLM backend URLs (default: auto-detect from config)")
    parser.add_argument("--config", default="/root/microllm/config.yaml", help="MicroLLM config.yaml path")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds")
    args = parser.parse_args()

    # Auto-detect from config
    backend_urls = []
    microllm_url = args.microllm

    if args.backends:
        backend_urls = args.backends
    else:
        detected, port = parse_microllm_config(args.config)
        if detected:
            backend_urls = detected
        else:
            backend_urls = ["http://localhost:8011"]

    if not microllm_url:
        try:
            _, port = parse_microllm_config(args.config)
            microllm_url = f"http://localhost:{port}"
        except Exception:
            microllm_url = "http://localhost:8012"

    console = Console()
    prev_backends = {}
    prev_time = None

    console.print(f"[bold]MicroLLM Monitor[/]")
    console.print(f"  MicroLLM:  {microllm_url}")
    console.print(f"  Backends:  {', '.join(backend_urls)}")
    console.print(f"  Interval:  {args.interval}s")
    console.print()
    time.sleep(1)

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            try:
                now = time.monotonic()
                dt = (now - prev_time) if prev_time else 0

                # Fetch all data
                microllm_stats, _ = fetch_json(f"{microllm_url}/stats")

                backend_metrics = {}
                for url in backend_urls:
                    m, _ = fetch_prometheus(f"{url}/metrics")
                    backend_metrics[url] = m

                gpus = fetch_nvidia_gpu()
                containers = fetch_podman_containers()

                state = {
                    "microllm_stats": microllm_stats,
                    "backend_metrics": backend_metrics,
                    "prev_backends": prev_backends,
                    "gpus": gpus,
                    "containers": containers,
                    "dt": dt,
                    "interval": args.interval,
                }

                live.update(build_dashboard(state))

                # Save snapshots for rate calculation
                new_prev = {}
                for url, bm in backend_metrics.items():
                    if bm:
                        new_prev[url] = {
                            "prompt": mv(bm, "vllm:prompt_tokens_total"),
                            "gen": mv(bm, "vllm:generation_tokens_total"),
                        }
                prev_backends = new_prev
                prev_time = now

                time.sleep(args.interval)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
