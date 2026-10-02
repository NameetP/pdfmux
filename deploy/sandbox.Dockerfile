# Parse-only image for the hosted ChatGPT plugin (src/pdfmux/remote/sandbox.py, PDFMUX_SANDBOX=docker).
# It is run with --network none --read-only --cap-drop ALL --user 65534 and memory/cpu/pid limits,
# and only ever executes `python -m pdfmux.remote.worker <file> <max_pages>`. No MCP server, no
# extras, no API keys: the worker needs PyMuPDF and pdfmux's pure-Python modules only.
#
# Build:  docker build -f deploy/sandbox.Dockerfile -t pdfmux-sandbox:latest .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src/ ./src/
RUN pip install --no-cache-dir . && rm -rf /app/src /root/.cache

USER 65534:65534
ENTRYPOINT []
CMD ["python", "-m", "pdfmux.remote.worker"]
