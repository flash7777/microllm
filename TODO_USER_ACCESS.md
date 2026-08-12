# User Access Mode für microllm

## Ziel

microllm soll Browser-User (chat-with-file Extension) bedienen können,
ohne separaten Proxy-Container. Server-Traffic (Taki, Worker) bleibt
unberührt.

## Dreistufiges Modell

```
Request rein
  └─ SRC-IP in server_ips?
       ├─ JA → Server-Mode: passthrough (wie bisher)
       └─ NEIN → API-Key im Header?
            ├─ JA + gültig → API-Key-Mode: passthrough (für Dienste mit eigenem Key)
            └─ NEIN → User-Mode: OIDC + Rate-Limit + Sanitize + Origin-Check
```

### 1. Server-Mode (IP-basiert)

Bekannte IPs (localhost, Pod-Netz). Kein Auth, kein Rate-Limit.
Für: Taki, OpenCloud-intern, Worker.

### 2. API-Key-Mode

Eigene microllm-API-Keys (nicht LLM-Backend-Keys!). Kein OIDC nötig,
aber Rate-Limit optional. Für: externe Dienste, CLI-Tools, Scripte.

### 3. User-Mode (OIDC)

Browser-Requests über Traefik. Voller Schutz:
- OIDC Token-Validierung (Bearer → oCIS userinfo)
- Per-User Rate-Limiting (sliding window, per `sub` claim)
- Origin-Validierung (CORS, allowed_origins)
- Request-Sanitierung (max_tokens clamp, body size limit)

## Config-Entwurf

```yaml
general_settings:
  port: 8012

  # Stufe 1: Server-IPs (passthrough, kein Auth)
  server_ips:
    - 127.0.0.1
    - 10.89.1.0/24

  # Stufe 2: API-Keys (passthrough, optional rate-limit)
  api_keys:
    - key: "mk-abc123..."
      name: "cli-tool"
      rate_limit: 30        # req/min, 0 = unlimited

  # Stufe 3: User-Mode (OIDC, voller Schutz)
  user_mode:
    enabled: true
    oidc_issuer: https://cloud.brandis.eu
    # -> GET /.well-known/openid-configuration -> userinfo_endpoint
    token_cache_ttl: 300    # userinfo-Response 5min cachen per Token
    rate_limit: 10          # req/min per User (sub claim)
    max_tokens: 4096        # clamp max_tokens in Request
    max_body_bytes: 1048576 # 1MB Body-Limit
    allowed_origins:
      - https://cloud.brandis.eu
```

## Implementierung

### Access-Log (DONE)

Client-IP wird bei jedem Request geloggt. Format:
```
  127.0.0.1        local-ocr             stream    25.3s  reqs=42  in=1200  out=800
```

Damit sammeln wir Daten für die server_ips Liste.

### IP-Check

Erster Check im Request-Handler (kein I/O):
- `ipaddress.ip_network` für CIDR-Matching
- Netzwerk-Objekte beim Config-Load parsen, nicht pro Request

### API-Key-Check

Header `Authorization: Bearer mk-...` oder `X-API-Key: mk-...`.
Prefix `mk-` unterscheidet microllm-Keys von OIDC-Tokens.
Dict-Lookup, O(1).

### OIDC-Validierung

1. Lazy: beim ersten User-Request OIDC Discovery fetchen
2. Bearer-Token aus Authorization-Header
3. GET userinfo_endpoint mit Bearer → 200 = valid, sub = User-ID
4. Response cachen (Token → sub + expiry), TTL aus Config
5. Cache: dict mit TTL-Pruning (kein Redis/externe Deps)

### Per-User Rate-Limiting

Sliding-Window-Counter pro `sub` claim:
- deque mit Timestamps pro User
- Alte Einträge (>60s) beim Check entfernen
- Über Limit → 429 Too Many Requests

### Request-Sanitierung (nur User-Mode)

- `max_tokens` auf Config-Wert deckeln
- Body-Size prüfen (vor JSON-Parse)
- Model muss in erlaubter Liste sein (optional: `user_models`)

### Origin-Check (nur User-Mode)

- `Origin` Header gegen `allowed_origins` prüfen
- CORS-Preflight (OPTIONS) beantworten
- Kein Origin-Header + nicht in server_ips → reject

## Reihenfolge

1. [x] Access-Log mit Client-IP
2. [ ] server_ips auswerten (ein paar Tage Daten sammeln)
3. [ ] IP-Check + API-Key-Check implementieren
4. [ ] OIDC + Rate-Limit + Sanitize implementieren
5. [ ] chat-with-file Extension integrieren
