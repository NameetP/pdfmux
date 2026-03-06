#!/usr/bin/env python3
"""Daily blog post generator for pdfmux.

Reads content-queue.json, picks the next highest-priority queued topic,
generates a full Hugo post via Claude API with real data enrichment,
runs quality checks, generates distribution drafts, and commits.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from enrich import build_enrichment_block

# -- Paths ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # /pdfmux/
BLOG_DIR = REPO_ROOT / "blog"
QUEUE_FILE = BLOG_DIR / "content-queue.json"
POSTS_DIR = BLOG_DIR / "content" / "posts"
DIST_DIR = BLOG_DIR / "distribution"
CONTEXT_FILE = Path(__file__).resolve().parent / "product-context.txt"

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 8192

# Known-safe URL prefixes. Posts add their own URLs dynamically.
BASE_ALLOWED_URLS = [
    "https://github.com/NameetP/pdfmux",
    "https://pypi.org/project/pdfmux",
    "https://pdfmux.com",
]

# CTA variants by content type
CTA_VARIANTS = {
    "tutorial": (
        "## Try it\n\n```bash\npip install pdfmux\npdfmux your-file.pdf\n```\n\n"
        "If this saved you time, "
        "[star us on GitHub](https://github.com/NameetP/pdfmux)."
    ),
    "benchmark": (
        "## Try the benchmark yourself\n\n```bash\npip install pdfmux\n"
        "pdfmux bench your-file.pdf\n```\n\n"
        "[GitHub](https://github.com/NameetP/pdfmux) — source, docs, examples. "
        "[PyPI](https://pypi.org/project/pdfmux/) — `pip install pdfmux`."
    ),
    "comparison": (
        "## Try it free\n\n```bash\npip install pdfmux\npdfmux your-file.pdf\n```\n\n"
        "MIT licensed. Runs locally. No API keys needed.\n\n"
        "[GitHub](https://github.com/NameetP/pdfmux) · "
        "[PyPI](https://pypi.org/project/pdfmux/)"
    ),
    "troubleshooting": (
        "## Fix it now\n\n```bash\npip install pdfmux\npdfmux your-file.pdf\n```\n\n"
        "Still stuck? "
        "[Open an issue](https://github.com/NameetP/pdfmux) — we respond fast."
    ),
    "architecture": (
        "## Dive in\n\n```bash\npip install pdfmux\npdfmux your-file.pdf\n```\n\n"
        "[Join the discussion on GitHub](https://github.com/NameetP/pdfmux). "
        "Contributions welcome."
    ),
}


# -- Queue helpers -----------------------------------------------------------

def load_queue() -> list[dict]:
    return json.loads(QUEUE_FILE.read_text())


def save_queue(queue: list[dict]) -> None:
    QUEUE_FILE.write_text(json.dumps(queue, indent=2) + "\n")


def pick_next_topic(queue: list[dict]) -> dict | None:
    """Pick the highest-priority queued topic, preferring earlier clusters."""
    cluster_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    queued = [e for e in queue if e["status"] == "queued"]
    if not queued:
        return None
    queued.sort(key=lambda e: (cluster_order.get(e.get("cluster", "D"), 4), e["priority"]))
    return queued[0]


def get_published_posts(queue: list[dict]) -> list[dict]:
    return [e for e in queue if e["status"] == "published"]


# -- Prompt builders ---------------------------------------------------------

def build_system_prompt(topic: dict, published: list[dict], enrichment: str) -> str:
    product_context = CONTEXT_FILE.read_text()

    # Build internal links section
    internal_links = ""
    if published:
        lines = [f"- https://pdfmux.com/blog/{p['slug']}/ ({p['title']})" for p in published]
        internal_links = "Link to these published posts where relevant:\n" + "\n".join(lines)

    # Build URL whitelist (base + published post URLs)
    url_lines = list(BASE_ALLOWED_URLS)
    for p in published:
        url_lines.append(f"https://pdfmux.com/blog/{p['slug']}/")
    url_whitelist = "\n".join(f"- {u}" for u in url_lines)

    # CTA variant
    cta = CTA_VARIANTS.get(topic.get("content_type", "tutorial"), CTA_VARIANTS["tutorial"])

    return f"""You are a technical blog writer for pdfmux, an open-source Python library for PDF extraction. You write developer-focused content that is casual, authoritative, and deeply technical.

## Product Context

{product_context}

{enrichment}

## Writing Style Guide

### Tone
- Casual-authoritative. Senior engineer explaining to a peer.
- First person for stories ("I built", "I tested").
- Second person for instructions ("you can", "your pipeline").
- No jargon: no "leverage", "utilize", "streamline", "empower".
- No unsupported superlatives. Use real numbers instead.
- Short sentences. Short paragraphs. One idea per paragraph.

### Structure (BLUF — Bottom Line Up Front)
- Start with a TL;DR paragraph (2-3 sentences) with the key finding.
- Follow with a horizontal rule (---).
- Use H2 (##) for major sections. Aim for 5-8 H2 sections.
- Use H3 (###) to nest under H2s. Never skip from H1 to H3.
- Put keywords in headings naturally.
- End with the CTA section below.

### Code Examples
- Every post MUST include at least one Python code example and one CLI example.
- Every post MUST include `pip install pdfmux` somewhere in the body.
- Use real pdfmux API calls. Code must be realistic and working.
- Fenced code blocks with language identifiers (```python, ```bash, ```json).
- Short snippets (3-10 lines), not walls of code.

### Tables
- Use markdown tables for comparisons and benchmark data.
- Always include a header row. Keep to 3-5 columns.

### SEO
- Meta description in frontmatter: 120-160 characters, includes primary keyword.
- Primary keyword in the first paragraph.
- Target keywords in at least 2 H2/H3 headings naturally.
- No keyword stuffing.

### Frontmatter (TOML)
Every post MUST start with this exact structure:
+++
date = 'YYYY-MM-DD'
draft = false
title = 'Post title here'
description = 'Meta description 120-160 chars'
tags = ['tag1', 'tag2']
slug = 'url-safe-slug'
+++

### URL Rules — CRITICAL
You may ONLY use these URLs. Do NOT invent or hallucinate any others:
{url_whitelist}

If you need to reference an external tool or concept, describe it without linking.

### Internal Links
{internal_links if internal_links else "No published posts yet. Skip internal linking."}

### CTA Section (use this exact pattern at the end of the post)
{cta}

### Word Count
Target {topic.get('word_count_target', 2000)} words. Quality over padding.

## Rules
- No author bylines or "About the author" sections
- No date in the body text (it's in frontmatter)
- No emojis in headings or body text
- No fictional benchmarks, user quotes, or case studies
- No features that pdfmux does not have
- No "in this article we will..." introductions. Get to the point.
- Output ONLY the markdown content starting with the +++ frontmatter. No preamble."""


def build_user_prompt(topic: dict) -> str:
    return f"""Write a complete blog post with the title: "{topic['title']}"

Target keywords to weave in naturally: {', '.join(topic['target_keywords'])}
Content type: {topic.get('content_type', 'tutorial')}
Target word count: {topic.get('word_count_target', 2000)}
Tags for frontmatter: {json.dumps(topic['tags'])}
Slug: {topic['slug']}

Additional context: {topic.get('notes', 'None')}"""


# -- Generation --------------------------------------------------------------

def generate_post(system_prompt: str, user_prompt: str) -> str:
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


# -- Frontmatter helpers -----------------------------------------------------

def fix_frontmatter(content: str, topic: dict, today: str) -> str:
    """Ensure frontmatter has correct values regardless of LLM output."""
    # Extract frontmatter block
    match = re.match(r"^\+\+\+\n(.*?)\n\+\+\+", content, re.DOTALL)
    if not match:
        # LLM didn't produce proper frontmatter — wrap it
        fm = (
            f"+++\ndate = '{today}'\ndraft = false\n"
            f"title = '{topic['title']}'\n"
            f"description = '{topic.get('description', '')}'\n"
            f"tags = {json.dumps(topic['tags'])}\n"
            f"slug = '{topic['slug']}'\n+++\n\n"
        )
        return fm + content

    fm_text = match.group(1)
    body = content[match.end():]

    # Force correct values
    def replace_or_add(text: str, key: str, value: str) -> str:
        pattern = rf"^{key}\s*=.*$"
        replacement = f"{key} = {value}"
        if re.search(pattern, text, re.MULTILINE):
            return re.sub(pattern, replacement, text, flags=re.MULTILINE)
        return text + f"\n{replacement}"

    fm_text = replace_or_add(fm_text, "date", f"'{today}'")
    fm_text = replace_or_add(fm_text, "draft", "false")
    fm_text = replace_or_add(fm_text, "slug", f"'{topic['slug']}'")
    fm_text = replace_or_add(fm_text, "tags", json.dumps(topic["tags"]))

    return f"+++\n{fm_text.strip()}\n+++{body}"


def extract_frontmatter_value(content: str, key: str) -> str:
    """Extract a value from TOML frontmatter."""
    match = re.search(rf"^{key}\s*=\s*'([^']*)'", content, re.MULTILINE)
    if match:
        return match.group(1)
    match = re.search(rf'^{key}\s*=\s*"([^"]*)"', content, re.MULTILINE)
    if match:
        return match.group(1)
    return ""


# -- Quality gates -----------------------------------------------------------

def quality_checks(content: str, topic: dict, published: list[dict]) -> list[str]:
    """Run 7 quality gates. Returns list of failures (empty = pass)."""
    failures = []

    # Strip frontmatter for word count
    body = re.sub(r"^\+\+\+\n.*?\n\+\+\+", "", content, flags=re.DOTALL).strip()
    word_count = len(body.split())

    # 1. Word count
    if not (1500 <= word_count <= 2500):
        failures.append(f"Word count {word_count} outside range 1500-2500")

    # 2. Code blocks
    if "```" not in content:
        failures.append("No code blocks found")

    # 3. Frontmatter completeness
    for field in ["date", "title", "description", "slug"]:
        val = extract_frontmatter_value(content, field)
        if not val:
            failures.append(f"Missing or empty frontmatter field: {field}")

    # Check tags exist
    if "tags = [" not in content and "tags = [" not in content:
        failures.append("Missing frontmatter field: tags")

    # Check draft = false
    if "draft = false" not in content:
        failures.append("draft is not set to false")

    # 4. No hallucinated URLs
    allowed_prefixes = list(BASE_ALLOWED_URLS)
    for p in published:
        if p.get("url"):
            allowed_prefixes.append(f"https://pdfmux.com{p['url']}")
    # Also allow the blog index
    allowed_prefixes.append("https://pdfmux.com/blog/")

    urls = re.findall(r"https?://[^\s)\]>\"'`]+", body)
    for url in urls:
        url_clean = url.rstrip(".,;:!?)")
        if not any(url_clean.startswith(prefix) for prefix in allowed_prefixes):
            failures.append(f"Hallucinated URL: {url_clean}")

    # 5. Slug is URL-safe
    slug = extract_frontmatter_value(content, "slug")
    if slug and not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", slug):
        failures.append(f"Slug '{slug}' is not URL-safe")

    # 6. Meta description length
    desc = extract_frontmatter_value(content, "description")
    if desc and not (120 <= len(desc) <= 160):
        failures.append(f"Meta description length {len(desc)} outside range 120-160")

    # 7. Contains pip install pdfmux
    if "pip install pdfmux" not in body:
        failures.append("Missing 'pip install pdfmux'")

    return failures


# -- Distribution draft generation -------------------------------------------

def generate_distribution(content: str, topic: dict) -> None:
    """Generate distribution drafts for 5 platforms."""
    slug = topic["slug"]
    dist_path = DIST_DIR / slug
    dist_path.mkdir(parents=True, exist_ok=True)

    blog_url = f"https://pdfmux.com/blog/{slug}/"
    title = topic["title"]
    desc = topic.get("description", "")
    tags = topic.get("tags", [])

    # Dev.to cross-post
    devto_tags = ", ".join(tags[:4])  # Dev.to allows max 4 tags
    devto = (
        f"---\ntitle: \"{title}\"\n"
        f"published: true\n"
        f"tags: {devto_tags}\n"
        f"canonical_url: {blog_url}\n---\n\n"
        f"*Originally published at [pdfmux.com/blog]({blog_url})*\n\n"
        f"{desc}\n\n"
        f"[Read the full post on pdfmux.com →]({blog_url})\n"
    )
    (dist_path / "devto.md").write_text(devto)

    # Twitter thread
    twitter = (
        f"🧵 Thread: {title}\n\n"
        f"---\n\n"
        f"Tweet 1 (Hook):\n{desc}\n\n"
        f"Tweet 2-5 (Key insights — extract from post):\n"
        f"[Fill from the post's main H2 sections]\n\n"
        f"Tweet 6 (CTA):\n"
        f"Full post with code examples and benchmarks:\n{blog_url}\n\n"
        f"pip install pdfmux\n"
    )
    (dist_path / "twitter.md").write_text(twitter)

    # Reddit r/Python
    reddit = (
        f"**Title:** {title}\n\n"
        f"**Subreddit:** r/Python (use 'Project Showcase' flair)\n\n"
        f"**Body:**\n\n"
        f"{desc}\n\n"
        f"```bash\npip install pdfmux\n```\n\n"
        f"[Key code example from the post goes here]\n\n"
        f"Full post with benchmarks and more examples: {blog_url}\n\n"
        f"MIT licensed. Source: https://github.com/NameetP/pdfmux\n\n"
        f"Feedback welcome!\n"
    )
    (dist_path / "reddit.md").write_text(reddit)

    # LinkedIn
    linkedin = (
        f"**LinkedIn Post (story format, no links in body):**\n\n"
        f"[Opening hook — 1-2 sentences about the problem this post solves]\n\n"
        f"[2-3 key insights from the post, one per short paragraph]\n\n"
        f"[Soft CTA: \"I wrote about this in detail on our blog.\"]\n\n"
        f"---\n**First comment (post link here):**\n{blog_url}\n"
    )
    (dist_path / "linkedin.md").write_text(linkedin)

    # Newsletter pitch
    newsletter = (
        f"**Subject:** {title}\n\n"
        f"**Pitch (2-3 sentences for Python Weekly / PyCoder's Weekly):**\n\n"
        f"{desc} Includes working code examples and real benchmark data.\n\n"
        f"{blog_url}\n"
    )
    (dist_path / "newsletter-pitch.md").write_text(newsletter)


# -- Git operations ----------------------------------------------------------

def write_and_commit(content: str, topic: dict, queue: list[dict]) -> None:
    """Write post, update queue, generate distribution, commit and push."""
    # Check for duplicate slug
    post_path = POSTS_DIR / f"{topic['slug']}.md"
    if post_path.exists():
        print(f"Post already exists: {post_path}. Skipping.")
        sys.exit(0)

    # Write the post
    post_path.write_text(content)

    # Update queue
    for entry in queue:
        if entry["id"] == topic["id"]:
            entry["status"] = "published"
            entry["published_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            entry["url"] = f"/blog/{topic['slug']}/"
            break
    save_queue(queue)

    # Generate distribution drafts
    generate_distribution(content, topic)

    # Git operations
    subprocess.run(
        ["git", "config", "user.name", "pdfmux-bot"],
        check=True, cwd=REPO_ROOT,
    )
    subprocess.run(
        ["git", "config", "user.email", "bot@pdfmux.com"],
        check=True, cwd=REPO_ROOT,
    )
    subprocess.run(
        ["git", "add", str(post_path), str(QUEUE_FILE),
         str(DIST_DIR / topic["slug"])],
        check=True, cwd=REPO_ROOT,
    )
    subprocess.run(
        ["git", "commit", "-m", f"blog: publish '{topic['title']}'"],
        check=True, cwd=REPO_ROOT,
    )
    subprocess.run(["git", "push"], check=True, cwd=REPO_ROOT)


# -- Main --------------------------------------------------------------------

def main() -> None:
    queue = load_queue()
    topic = pick_next_topic(queue)

    if topic is None:
        print("No queued topics remaining. Exiting.")
        sys.exit(0)

    published = get_published_posts(queue)
    print(f"Generating post: {topic['title']}")
    print(f"  Cluster: {topic.get('cluster', '?')} | Priority: {topic['priority']}")
    print(f"  Keywords: {', '.join(topic['target_keywords'])}")

    # Step 1: Data enrichment
    print("Step 1: Enriching with real data...")
    enrichment = build_enrichment_block()
    print(enrichment)

    # Step 2: Generate
    print("Step 2: Generating post via Claude API...")
    system_prompt = build_system_prompt(topic, published, enrichment)
    user_prompt = build_user_prompt(topic)
    content = generate_post(system_prompt, user_prompt)

    # Fix frontmatter
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    content = fix_frontmatter(content, topic, today)

    # Step 3: Quality gates
    print("Step 3: Running quality checks...")
    failures = quality_checks(content, topic, published)
    if failures:
        print("Quality checks FAILED:")
        for f in failures:
            print(f"  ✗ {f}")
        sys.exit(1)
    print("  All 7 quality gates passed.")

    # Step 4: Publish
    print("Step 4: Writing files and committing...")
    write_and_commit(content, topic, queue)
    print(f"Published: /blog/{topic['slug']}/")

    # Step 5: Distribution drafts
    print(f"Step 5: Distribution drafts at blog/distribution/{topic['slug']}/")
    print("Done.")


if __name__ == "__main__":
    main()
