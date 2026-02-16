# microllm

Leichtgewichtiger Anthropic-API-Proxy mit Backend-Routing und OCR-Integration.

## Features

- **Anthropic API** (`/v1/messages`) → OpenAI-Backend (vLLM, llama.cpp)
- **Model-Routing**: Beliebige Model-Namen auf Backends mappen
- **Auto-Discovery**: Erkennt Backend-Modelle automatisch beim Start
- **OCR-Integration**: PDF-Dokumente via DeepSeek-OCR-2 zu Markdown

## Starten

```bash
./start
```

## Endpoints

| Endpoint | Beschreibung |
|----------|--------------|
| `GET /health` | Health-Check |
| `GET /stats` | Request-Statistiken |
| `GET /v1/models` | Verfügbare Modelle |
| `POST /v1/messages` | Anthropic Messages API |

## Auto-Discovery

Beim Start fragt microllm alle konfigurierten Backends ab (`/v1/models`) und:
1. Erkennt den tatsächlichen Model-Namen
2. Speichert `max_model_len` für Validierung
3. Erstellt Aliases für gefundene Modelle

**Wichtig:** Nach Backend-Änderungen (z.B. neues Modell in vLLM) muss microllm neu gestartet werden:

```bash
cd ~/microllm && ./start
```

## Konfiguration

`config.yaml`:

```yaml
general_settings:
  ocr_url: http://localhost:8019      # DeepSeek-OCR-2 Service
  chatlog_dir: /tmp/microllm-chatlog

model_list:
  - model_name: local                  # Alias für Clients
    litellm_params:
      model: hosted_vllm/glm-4.7-flash # Backend-Modell
      api_base: http://0.0.0.0:8011/v1
      api_key: dummy-key
      max_model_len: 100000

  # Claude Code Model-Namen → lokales GLM
  - model_name: haiku
    litellm_params:
      model: hosted_vllm/glm-4.7-flash
      api_base: http://0.0.0.0:8011/v1
      ...
```

## OCR-Integration

Wenn ein Request `type: "document"` Blöcke enthält (PDFs):
1. microllm extrahiert Base64-PDF
2. Sendet an OCR-Service (`ocr_url`)
3. Ersetzt document-Block durch text-Block mit Markdown

## Aktuelle Backends

| Backend | Port | Modell |
|---------|------|--------|
| vLLM (GPU) | 8011 | glm-4.7-flash (FP8, 40 tok/s) |
| llama.cpp (CPU) | 8018 | glm-4.7 (Q6_K) |
| DeepSeek-OCR-2 | 8019 | OCR für PDFs |

## Logs

```bash
podman logs -f microllm
```
