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

## Systemd-Service (rootless Podman)

Einmalige Einrichtung:

```bash
# 1. System-User anlegen
sudo useradd -r -s /usr/sbin/nologin -m -d /home/microllm microllm
sudo loginctl enable-linger microllm

# 2. subuid/subgid (rootless Podman)
echo "microllm:200000:65536" | sudo tee -a /etc/subuid
echo "microllm:200000:65536" | sudo tee -a /etc/subgid

# 3. Storage-Config (ZFS-Hosts brauchen fuse-overlayfs)
sudo -u microllm mkdir -p /home/microllm/.config/containers
cat <<'CONF' | sudo tee /home/microllm/.config/containers/storage.conf
[storage]
driver = "overlay"
[storage.options.overlay]
mount_program = "/usr/bin/fuse-overlayfs"
CONF

# 4. Image + Config kopieren
podman save localhost/microllm:latest -o /tmp/microllm-image.tar
sudo -u microllm XDG_RUNTIME_DIR=/run/user/$(id -u microllm) podman load -i /tmp/microllm-image.tar
sudo cp config.yaml /home/microllm/microllm/config.yaml
sudo chown -R microllm:microllm /home/microllm/

# 5. Service installieren
sudo -u microllm mkdir -p /home/microllm/.config/systemd/user
sudo cp microllm.service /home/microllm/.config/systemd/user/
sudo chown -R microllm:microllm /home/microllm/.config
sudo -u microllm XDG_RUNTIME_DIR=/run/user/$(id -u microllm) systemctl --user daemon-reload
sudo -u microllm XDG_RUNTIME_DIR=/run/user/$(id -u microllm) systemctl --user enable --now microllm
```

Service-Management:

```bash
# Status / Logs / Restart
sudo -u microllm XDG_RUNTIME_DIR=/run/user/$(id -u microllm) systemctl --user status microllm
sudo -u microllm XDG_RUNTIME_DIR=/run/user/$(id -u microllm) journalctl --user -u microllm -f
sudo -u microllm XDG_RUNTIME_DIR=/run/user/$(id -u microllm) systemctl --user restart microllm
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
