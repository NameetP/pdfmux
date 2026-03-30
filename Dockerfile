FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir build && \
    python -m build --wheel

FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /app/dist/*.whl /tmp/

# Install pdfmux with MCP server + HTTP transport dependencies
RUN pip install --no-cache-dir /tmp/*.whl && \
    pip install --no-cache-dir "mcp>=1.0.0" "uvicorn>=0.30.0" && \
    rm /tmp/*.whl

# Default: HTTP transport for Smithery deployment
ENV TRANSPORT=http
ENV PDFMUX_ALLOWED_DIRS=/tmp
EXPOSE 8000

ENTRYPOINT ["pdfmux"]
CMD ["serve"]
