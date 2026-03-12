#!/usr/bin/env python3
"""Design system linter — enforces pdfmux visual rules on HTML/CSS files.

Usage:
    python design-system/lint.py site/              # lint all HTML files in site/
    python design-system/lint.py site/index.html    # lint one file
    python design-system/lint.py --staged           # lint only git-staged site/ files

Exit codes:
    0  — no violations
    1  — violations found (prints report)

Checks:
    1. No raw hex colors outside :root and [data-theme="dark"]
    2. No raw rgb/hsl colors outside :root and [data-theme="dark"] (shadows allowed)
    3. No inline font-family (must use var(--font) or var(--mono))
    4. No arbitrary border-radius (must be 4px, 8px, or 10px)
    5. No transition durations other than 0.15s
    6. No !important
    7. No external stylesheets (<link rel="stylesheet">)
    8. No external JS (<script src="...">)
    9. Max-width must not exceed 720px
    10. No emoji icons in feature cards
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Violation tracking
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    file: str
    line: int
    rule: str
    message: str
    snippet: str

    def __str__(self) -> str:
        return f"  {self.file}:{self.line}  [{self.rule}]  {self.message}\n    {self.snippet.strip()}"


# ---------------------------------------------------------------------------
# Allowed values
# ---------------------------------------------------------------------------

ALLOWED_RADII = {"4px", "8px", "10px"}
ALLOWED_TRANSITION_DURATIONS = {"0.15s"}
ALLOWED_MAX_WIDTHS = {"720px"}

# Spacing scale — these are the only pixel values allowed for margin/padding
ALLOWED_SPACING = {
    "0", "0px",
    "2px", "3px", "4px", "6px", "8px", "10px", "12px", "14px",
    "16px", "20px", "22px", "24px", "28px", "48px", "56px", "60px", "80px",
}

# Colors that may appear as raw values (only inside :root / [data-theme="dark"])
HEX_PATTERN = re.compile(r"#[0-9a-fA-F]{3,8}\b")
RGB_PATTERN = re.compile(r"\brgba?\s*\(")
HSL_PATTERN = re.compile(r"\bhsla?\s*\(")

# Patterns for rules
FONT_FAMILY_PATTERN = re.compile(r"font-family\s*:", re.IGNORECASE)
BORDER_RADIUS_PATTERN = re.compile(r"border-radius\s*:\s*([^;]+)")
TRANSITION_PATTERN = re.compile(r"transition\s*:[^;]*?(\d+\.?\d*s)")
IMPORTANT_PATTERN = re.compile(r"!important")
EXTERNAL_CSS_PATTERN = re.compile(r'<link[^>]*rel\s*=\s*["\']stylesheet["\']', re.IGNORECASE)
EXTERNAL_JS_PATTERN = re.compile(r'<script[^>]*src\s*=', re.IGNORECASE)
MAX_WIDTH_PATTERN = re.compile(r"max-width\s*:\s*(\d+px)")

# Emoji detection — Unicode emoji ranges commonly used as icons
EMOJI_PATTERN = re.compile(
    r'[\U0001F300-\U0001F9FF'   # Misc symbols, emoticons, symbols & pictographs
    r'\U00002600-\U000027BF'     # Misc symbols, dingbats
    r'\U0000FE00-\U0000FE0F'     # Variation selectors
    r'\U0001FA00-\U0001FA6F'     # Chess symbols
    r'\U0001FA70-\U0001FAFF'     # Symbols extended-A
    r'\U0000200D'                 # Zero width joiner
    r'\U00002702-\U000027B0]'    # Dingbats
)

# CSS property patterns for spacing check
SPACING_PROPS = re.compile(
    r"(?:margin|padding|gap|top|right|bottom|left)"
    r"(?:-(?:top|right|bottom|left))?\s*:\s*([^;]+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------

def is_inside_token_block(lines: list[str], line_idx: int) -> bool:
    """Check if a line is inside a :root { ... } or [data-theme="dark"] { ... } block.

    These are the only places where raw hex/rgb/hsl values are allowed.
    """
    brace_depth = 0
    in_block = False
    for i in range(line_idx):
        line = lines[i]
        if (":root" in line or 'data-theme="dark"' in line or "data-theme='dark'" in line) and "{" in line:
            in_block = True
            brace_depth = 1
            continue
        if in_block:
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                in_block = False
    # Check the target line itself
    if in_block:
        brace_depth += lines[line_idx].count("{") - lines[line_idx].count("}")
        return brace_depth > 0
    return False


# Keep old name as alias for backward compatibility
is_inside_root = is_inside_token_block


def is_inside_style(lines: list[str], line_idx: int) -> bool:
    """Check if a line is inside a <style> block."""
    in_style = False
    for i in range(line_idx):
        if "<style" in lines[i]:
            in_style = True
        if "</style>" in lines[i]:
            in_style = False
    return in_style


def is_inside_html_body(lines: list[str], line_idx: int) -> bool:
    """Check if a line is inside the HTML <body> (not inside <style> or <script>)."""
    in_body = False
    in_style = False
    in_script = False
    for i in range(line_idx):
        if "<body" in lines[i]:
            in_body = True
        if "</body>" in lines[i]:
            in_body = False
        if "<style" in lines[i]:
            in_style = True
        if "</style>" in lines[i]:
            in_style = False
        if "<script" in lines[i]:
            in_script = True
        if "</script>" in lines[i]:
            in_script = False
    return in_body and not in_style and not in_script


def is_comment_line(line: str) -> bool:
    """Check if a line is a CSS or HTML comment."""
    stripped = line.strip()
    return stripped.startswith("/*") or stripped.startswith("<!--") or stripped.startswith("//")


# ---------------------------------------------------------------------------
# Lint checks
# ---------------------------------------------------------------------------

def check_raw_colors(lines: list[str], file_path: str) -> list[Violation]:
    """Rule: no raw hex/rgb/hsl colors outside :root and [data-theme='dark']."""
    violations = []
    for i, line in enumerate(lines):
        if is_comment_line(line):
            continue
        if not is_inside_style(lines, i):
            continue

        # Skip lines inside :root or [data-theme="dark"] — that's where tokens are defined
        if is_inside_token_block(lines, i):
            continue

        # Check for raw hex colors
        # Allow hex in data URIs and SVG inline content
        if "data:image" in line or "xmlns=" in line:
            continue

        # Allow rgba() in shadow/box-shadow definitions (design system shadows use rgba)
        if "shadow" in line.lower() and RGB_PATTERN.search(line):
            pass  # Don't flag rgb in shadow properties
        elif RGB_PATTERN.search(line):
            violations.append(Violation(
                file=file_path,
                line=i + 1,
                rule="color/no-raw-rgb",
                message="Raw rgb() color — use a CSS custom property instead",
                snippet=line,
            ))

        for match in HEX_PATTERN.finditer(line):
            # Allow #fff and #ffffff for button text, #4338ca and #6366f1 for hover states
            hex_val = match.group().lower()
            if hex_val in ("#fff", "#ffffff", "#4338ca", "#6366f1"):
                continue
            violations.append(Violation(
                file=file_path,
                line=i + 1,
                rule="color/no-raw-hex",
                message=f"Raw hex color `{match.group()}` — use a CSS custom property instead",
                snippet=line,
            ))

        if HSL_PATTERN.search(line):
            violations.append(Violation(
                file=file_path,
                line=i + 1,
                rule="color/no-raw-hsl",
                message="Raw hsl() color — use a CSS custom property instead",
                snippet=line,
            ))

    return violations


def check_font_family(lines: list[str], file_path: str) -> list[Violation]:
    """Rule: no inline font-family — must use var(--font) or var(--mono)."""
    violations = []
    for i, line in enumerate(lines):
        if is_comment_line(line):
            continue
        if not is_inside_style(lines, i):
            continue
        if is_inside_token_block(lines, i):
            continue

        if FONT_FAMILY_PATTERN.search(line):
            # Allow if using var()
            if "var(--font)" in line or "var(--mono)" in line:
                continue
            violations.append(Violation(
                file=file_path,
                line=i + 1,
                rule="typography/no-inline-font",
                message="Inline font-family — use `var(--font)` or `var(--mono)`",
                snippet=line,
            ))

    return violations


def check_border_radius(lines: list[str], file_path: str) -> list[Violation]:
    """Rule: border-radius must be 4px, 8px, or 10px."""
    violations = []
    for i, line in enumerate(lines):
        if is_comment_line(line):
            continue
        if not is_inside_style(lines, i):
            continue

        match = BORDER_RADIUS_PATTERN.search(line)
        if match:
            value = match.group(1).strip().rstrip(";")
            # Handle shorthand (e.g., "10px 10px 0 0")
            parts = value.split()
            for part in parts:
                part = part.strip()
                if part and part not in ALLOWED_RADII and part != "0":
                    violations.append(Violation(
                        file=file_path,
                        line=i + 1,
                        rule="spacing/border-radius",
                        message=f"Border-radius `{part}` not in allowed set ({', '.join(sorted(ALLOWED_RADII))})",
                        snippet=line,
                    ))

    return violations


def check_transitions(lines: list[str], file_path: str) -> list[Violation]:
    """Rule: all transition durations must be 0.15s."""
    violations = []
    for i, line in enumerate(lines):
        if is_comment_line(line):
            continue
        if not is_inside_style(lines, i):
            continue

        for match in TRANSITION_PATTERN.finditer(line):
            duration = match.group(1)
            if duration not in ALLOWED_TRANSITION_DURATIONS:
                violations.append(Violation(
                    file=file_path,
                    line=i + 1,
                    rule="transition/duration",
                    message=f"Transition duration `{duration}` — must be `0.15s`",
                    snippet=line,
                ))

    return violations


def check_important(lines: list[str], file_path: str) -> list[Violation]:
    """Rule: no !important."""
    violations = []
    for i, line in enumerate(lines):
        if is_comment_line(line):
            continue
        if IMPORTANT_PATTERN.search(line):
            violations.append(Violation(
                file=file_path,
                line=i + 1,
                rule="specificity/no-important",
                message="`!important` is not allowed — fix selector specificity instead",
                snippet=line,
            ))
    return violations


def check_external_deps(lines: list[str], file_path: str) -> list[Violation]:
    """Rule: no external CSS or JS."""
    violations = []
    for i, line in enumerate(lines):
        if EXTERNAL_CSS_PATTERN.search(line):
            violations.append(Violation(
                file=file_path,
                line=i + 1,
                rule="architecture/no-external-css",
                message="External stylesheet — site must be self-contained",
                snippet=line,
            ))
        if EXTERNAL_JS_PATTERN.search(line):
            # Allow self-hosted analytics (Umami)
            if "analytics.pdfmux.com" in line:
                continue
            violations.append(Violation(
                file=file_path,
                line=i + 1,
                rule="architecture/no-external-js",
                message="External JS — use inline vanilla JS only",
                snippet=line,
            ))
    return violations


def check_max_width(lines: list[str], file_path: str) -> list[Violation]:
    """Rule: max-width must not exceed 720px."""
    violations = []
    for i, line in enumerate(lines):
        if is_comment_line(line):
            continue
        match = MAX_WIDTH_PATTERN.search(line)
        if match:
            value = int(match.group(1).replace("px", ""))
            if value > 720:
                violations.append(Violation(
                    file=file_path,
                    line=i + 1,
                    rule="layout/max-width",
                    message=f"Max-width `{value}px` exceeds 720px limit",
                    snippet=line,
                ))
    return violations


def check_emoji_icons(lines: list[str], file_path: str) -> list[Violation]:
    """Rule: no emoji icons in feature cards — use inline SVG."""
    violations = []
    in_feature = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Track if we're inside a .feature div
        if 'class="feature"' in line or "class='feature'" in line:
            in_feature = True
        if in_feature and stripped == "</div>" and not any(
            tag in line for tag in ["</span>", "</code>", "</a>"]
        ):
            # Simple heuristic: a standalone </div> might close the feature
            pass

        # Check for emoji in icon spans within feature cards
        if 'class="icon"' in line or "class='icon'" in line:
            if EMOJI_PATTERN.search(line):
                violations.append(Violation(
                    file=file_path,
                    line=i + 1,
                    rule="icons/no-emoji",
                    message="Emoji icon in feature card — use inline SVG with `currentColor`",
                    snippet=line,
                ))

        # Also check HTML entities that decode to emoji (&#x1fXXX;)
        if 'class="icon"' in line or "class='icon'" in line:
            if re.search(r"&#x[0-9a-fA-F]+;", line) and "<svg" not in line:
                violations.append(Violation(
                    file=file_path,
                    line=i + 1,
                    rule="icons/no-emoji",
                    message="Emoji HTML entity in feature card — use inline SVG with `currentColor`",
                    snippet=line,
                ))

    return violations


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def lint_file(file_path: Path) -> list[Violation]:
    """Run all design system checks on a single file."""
    text = file_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    file_str = str(file_path)

    violations = []
    violations.extend(check_raw_colors(lines, file_str))
    violations.extend(check_font_family(lines, file_str))
    violations.extend(check_border_radius(lines, file_str))
    violations.extend(check_transitions(lines, file_str))
    violations.extend(check_important(lines, file_str))
    violations.extend(check_external_deps(lines, file_str))
    violations.extend(check_max_width(lines, file_str))
    violations.extend(check_emoji_icons(lines, file_str))

    return violations


def get_staged_files() -> list[Path]:
    """Get git-staged HTML files in site/."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True, text=True, check=True,
        )
        files = []
        for line in result.stdout.strip().split("\n"):
            if line and line.startswith("site/") and line.endswith(".html"):
                files.append(Path(line))
        return files
    except subprocess.CalledProcessError:
        return []


def main() -> int:
    args = sys.argv[1:]

    if not args:
        print("Usage: python design-system/lint.py site/")
        print("       python design-system/lint.py site/index.html")
        print("       python design-system/lint.py --staged")
        return 2

    if "--staged" in args:
        files = get_staged_files()
        if not files:
            print("No staged site/ HTML files to lint.")
            return 0
    else:
        target = Path(args[0])
        if target.is_dir():
            files = list(target.glob("**/*.html"))
        elif target.is_file():
            files = [target]
        else:
            print(f"Not found: {target}")
            return 2

    all_violations: list[Violation] = []
    for f in sorted(files):
        violations = lint_file(f)
        all_violations.extend(violations)

    if all_violations:
        print(f"\n{'='*60}")
        print(f"  DESIGN SYSTEM: {len(all_violations)} violation(s) found")
        print(f"{'='*60}\n")
        for v in all_violations:
            print(v)
            print()
        print(f"Fix these violations before committing.")
        print(f"Rules: design-system/DESIGN_SYSTEM.md")
        return 1
    else:
        print(f"Design system: {len(files)} file(s) checked, no violations.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
