FROM docker.io/library/python:3.12-slim

RUN pip install --no-cache-dir aiohttp pyyaml

WORKDIR /app
COPY microllm.py .

ENTRYPOINT ["python3", "microllm.py"]
CMD ["/config/config.yaml", "8012"]
