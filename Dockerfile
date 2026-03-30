FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir build && \
    python -m build --wheel

FROM python:3.12-slim

LABEL org.opencontainers.image.title="pdfmux MCP Server"
LABEL org.opencontainers.image.description="PDF extraction that actually works. MCP server for AI agents."
LABEL org.opencontainers.image.url="https://pdfmux.com"
LABEL org.opencontainers.image.source="https://github.com/NameetP/pdfmux"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app
COPY --from=builder /app/dist/*.whl /tmp/

# Install pdfmux with MCP + tables + OCR for maximum capability
RUN pip install --no-cache-dir /tmp/*.whl && \
    pip install --no-cache-dir "mcp>=1.0.0" "uvicorn>=0.30.0" && \
    rm /tmp/*.whl

# Default: HTTP transport for Smithery / remote deployment
ENV TRANSPORT=http
ENV PDFMUX_ALLOWED_DIRS=/tmp
EXPOSE 8000

# Health check for container orchestrators
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/mcp')" || exit 1

ENTRYPOINT ["pdfmux"]
CMD ["serve"]
