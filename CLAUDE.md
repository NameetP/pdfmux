# pdfmux — Agent Instructions

## Project

PDF extraction that checks its own work. Multi-pass pipeline: fast extract → audit → selective OCR → merge.

- **Language**: Python 3.11+
- **Build**: hatchling (pyproject.toml)
- **Lint**: ruff
- **Test**: pytest
- **Site**: static HTML in `site/` (no build step)

## Commands

```bash
# Python
ruff check src/ tests/          # lint
ruff format src/ tests/         # format
pytest -v                       # test

# Site
python3 design-system/lint.py site/   # design system lint
npx serve site/ -l 3456              # local dev server
```

## Design System

**When editing any file in `site/`, you MUST follow the design system.**

1. Read `design-system/DESIGN_SYSTEM.md` for all visual rules
2. Read `design-system/SKILL.md` for agent-specific instructions
3. After changes, run `python3 design-system/lint.py site/` and fix all violations

Key rules:
- No raw hex/rgb/hsl colors outside `:root` — use CSS custom properties
- No inline `font-family` — use `var(--font)` or `var(--mono)`
- Border-radius: only `4px`, `8px`, or `10px`
- Transitions: only `0.15s` duration
- No external CSS or JS — site is self-contained HTML
- Max content width: `720px`
