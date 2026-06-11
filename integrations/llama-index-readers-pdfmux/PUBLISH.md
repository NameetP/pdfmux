# Publishing `llama-index-readers-pdfmux`

LlamaIndex no longer accepts new integration packages into the `run-llama/llama_index` monorepo (new `pyproject.toml` PRs are auto-closed). The supported path is: **maintain this package here, publish independently to PyPI.** Once published, any LlamaIndex user can:

```bash
pip install llama-index-readers-pdfmux
```
```python
from llama_index.readers.pdfmux import PDFMuxReader
```

## Build + publish

```bash
cd integrations/llama-index-readers-pdfmux
python -m build                 # produces dist/*.whl + dist/*.tar.gz
twine upload dist/*             # needs a PyPI token for the account that owns pdfmux
```

## Verify before publish

```bash
python -m pytest tests/         # 4 tests; mocks pdfmux.load_llm_context
```

## Versioning

Bump `version` in `pyproject.toml` per release. `0.1.0` is the initial publish.

## Discovery (after first publish)

- The package name follows the `llama-index-readers-*` convention, so it surfaces in PyPI search for LlamaIndex readers.
- LlamaHub (llamahub.ai) indexes the `[tool.llamahub]` block in `pyproject.toml`; check whether the current LlamaHub registry accepts a listing entry for PyPI-published integrations and add one if so.
