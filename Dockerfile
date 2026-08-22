FROM docker.io/library/python:3.12-slim

# curl für Live-Diagnose im Container (Webtooling-Tests, Egress-Prüfung)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists

RUN pip install --no-cache-dir aiohttp pyyaml

WORKDIR /app
COPY microllm.py .

ENTRYPOINT ["python3", "microllm.py"]
CMD ["/config/config.yaml", "8012"]
