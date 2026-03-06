#!/usr/bin/env python3
"""Weekly keyword analysis for pdfmux blog.

Loads the content queue, published posts, events calendar, and GitHub issues,
then asks Claude to reprioritize the queue and suggest 0-3 new topics.
Commits the updated queue and writes an analysis report.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

# -- Paths ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # /pdfmux/
BLOG_DIR = REPO_ROOT / "blog"
QUEUE_FILE = BLOG_DIR / "content-queue.json"
POSTS_DIR = BLOG_DIR / "content" / "posts"
EVENTS_FILE = BLOG_DIR / "events-calendar.json"
COMPETITORS_FILE = BLOG_DIR / "competitors.json"
REPORT_DIR = BLOG_DIR / "analysis-reports"

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096


# -- Data loaders -----------------------------------------------------------

def load_queue() -> list[dict]:
    return json.loads(QUEUE_FILE.read_text())


def save_queue(queue: list[dict]) -> None:
    QUEUE_FILE.write_text(json.dumps(queue, indent=2) + "\n")


def get_published_titles() -> list[str]:
    """Get titles of all published posts from content/posts/."""
    titles = []
    if POSTS_DIR.exists():
        for md in POSTS_DIR.glob("*.md"):
            text = md.read_text()
            match = re.search(r"^title\s*=\s*['\"](.+?)['\"]", text, re.MULTILINE)
            if match:
                titles.append(match.group(1))
    return titles


def load_events_calendar() -> list[dict]:
    if EVENTS_FILE.exists():
        return json.loads(EVENTS_FILE.read_text())
    return []


def load_competitors() -> list[dict]:
    if COMPETITORS_FILE.exists():
        data = json.loads(COMPETITORS_FILE.read_text())
        return data.get("competitors", [])
    return []


def get_github_issues() -> str:
    """Pull recent GitHub issues that might inspire content."""
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--repo", "NameetP/pdfmux",
             "--limit", "10", "--json", "title,labels,url",
             "--jq", '.[] | "- \\(.title) (\\([.labels[].name] | join(", ")))"'],
            capture_output=True, text=True, timeout=15,
        )
        return result.stdout.strip() if result.returncode == 0 else "No issues available"
    except Exception:
        return "GitHub CLI not available"


def get_upcoming_events(events: list[dict]) -> list[dict]:
    """Filter events happening in the next 4 weeks."""
    now = datetime.now(timezone.utc)
    current_month = now.month
    # Show events in current month or next month
    return [e for e in events if e["month"] in (current_month, (current_month % 12) + 1)]


# -- Prompt builders --------------------------------------------------------

P0_KEYWORDS = [
    "pdf to markdown python", "extract text from pdf python",
    "pdf extraction python", "mcp server pdf", "pdf parser python",
    "langchain pdf loader", "rag pdf processing", "pdf ocr python",
    "scanned pdf to text python", "extract tables from pdf python",
    "pdf to text python library", "self-healing pdf extraction",
    "pdf confidence scoring python", "pdf extraction accuracy comparison",
    "pdf extraction not working python", "best python library for pdf extraction 2026",
    "unstructured io alternative", "llamaparse alternative",
    "chunking pdf for rag", "pdf for rag pipeline",
    "ai agent pdf extraction", "llm pdf processing",
    "python pdf pipeline", "best pdf parser for rag",
    "which pdf extractor to use python", "ocr accuracy python improve",
    "pdf extraction wrong text order", "pdf table extraction broken",
    "claude pdf processing", "open source pdf python",
]

CLUSTER_SUMMARY = """
Cluster A: Pipeline fundamentals — self-healing, confidence, benchmarks
Cluster B: Integrations — LangChain, LlamaIndex, MCP
Cluster C: Document types — tables, scanned, mixed, chunking
Cluster D: Comparisons — vs competitors, which tool guide, annual roundups
"""


def build_analysis_prompt(
    queue: list[dict],
    published_titles: list[str],
    events: list[dict],
    upcoming_events: list[dict],
    issues: str,
    competitors: list[dict],
) -> str:
    queue_json = json.dumps(queue, indent=2)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    upcoming_str = ""
    if upcoming_events:
        lines = [f"- {e['event']} (month {e['month']}): {e['content_play']}" for e in upcoming_events]
        upcoming_str = "UPCOMING EVENTS (next 4 weeks):\n" + "\n".join(lines)

    competitor_str = ""
    if competitors:
        lines = [f"- {c['name']}: {c['description']}" for c in competitors]
        competitor_str = "COMPETITORS:\n" + "\n".join(lines)

    return f"""You are an expert SEO content strategist for pdfmux, an open-source Python PDF extraction library.

Today: {today}

## Current Content Queue
{queue_json}

## Already Published
{chr(10).join(f'- {t}' for t in published_titles) if published_titles else 'No posts published yet.'}

## P0 Keywords (30 target keywords)
{chr(10).join(f'- {kw}' for kw in P0_KEYWORDS)}

## Topical Clusters
{CLUSTER_SUMMARY}

## GitHub Issues (potential content ideas)
{issues}

{upcoming_str}

{competitor_str}

## Your Task

Analyze the content queue and recommend changes. Return STRICT JSON with this schema:

{{
  "report": "2-3 paragraph analysis of current strategy, gaps, and recommendations",
  "priority_changes": [
    {{"id": "post-NNN", "new_priority": N, "reason": "why"}}
  ],
  "new_topics": [
    {{
      "title": "Post title",
      "slug": "url-safe-slug",
      "description": "Meta description 120-160 chars",
      "target_keywords": ["kw1", "kw2"],
      "tags": ["tag1", "tag2"],
      "cluster": "A|B|C|D",
      "content_type": "tutorial|benchmark|comparison|troubleshooting|architecture",
      "word_count_target": 2000,
      "evergreen": true,
      "notes": "Why this topic now"
    }}
  ],
  "cluster_recommendation": "Which cluster to focus on next week and why"
}}

Rules:
- Max 3 new topics. Only add if there's a real gap or timely opportunity.
- Priority changes should be based on upcoming events, keyword gaps, or cluster strategy.
- If an event is upcoming, boost related topics by lowering their priority number.
- Don't duplicate existing topics. Check slugs carefully.
- New topic slugs must be URL-safe: lowercase, hyphens only, no special chars.
- Output ONLY the JSON. No markdown fences, no explanation outside the JSON."""


# -- Analysis ---------------------------------------------------------------

def run_analysis(prompt: str) -> dict:
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text.strip()
    # Strip markdown fences if present
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def apply_changes(queue: list[dict], result: dict) -> tuple[list[dict], str]:
    """Apply priority changes and new topics. Returns (updated_queue, change_log)."""
    changes = []

    # Apply priority changes
    for change in result.get("priority_changes", []):
        for entry in queue:
            if entry["id"] == change["id"] and entry["status"] == "queued":
                old = entry["priority"]
                entry["priority"] = change["new_priority"]
                changes.append(f"  {entry['id']}: priority {old} → {change['new_priority']} ({change['reason']})")
                break

    # Add new topics
    existing_slugs = {e["slug"] for e in queue}
    next_id = max(int(e["id"].split("-")[1]) for e in queue) + 1

    for topic in result.get("new_topics", [])[:3]:  # Cap at 3
        if topic["slug"] in existing_slugs:
            changes.append(f"  Skipped '{topic['slug']}' — slug already exists")
            continue

        new_entry = {
            "id": f"post-{next_id:03d}",
            "title": topic["title"],
            "slug": topic["slug"],
            "description": topic.get("description", ""),
            "target_keywords": topic.get("target_keywords", []),
            "tags": topic.get("tags", []),
            "priority": next_id,  # Add at end by default
            "phase": 4,  # Added by analyzer
            "cluster": topic.get("cluster", "A"),
            "status": "queued",
            "content_type": topic.get("content_type", "tutorial"),
            "word_count_target": topic.get("word_count_target", 2000),
            "evergreen": topic.get("evergreen", True),
            "published_date": None,
            "url": None,
            "added_by": "keyword-analyzer",
            "added_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "notes": topic.get("notes", "Added by weekly keyword analysis"),
        }
        queue.append(new_entry)
        existing_slugs.add(topic["slug"])
        changes.append(f"  Added: {new_entry['id']} — {topic['title']}")
        next_id += 1

    return queue, "\n".join(changes) if changes else "  No changes."


def write_report(result: dict, change_log: str) -> Path:
    """Write analysis report to blog/analysis-reports/."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_path = REPORT_DIR / f"{today}-analysis.md"

    report = f"""# Keyword Analysis Report — {today}

## Analysis

{result.get('report', 'No analysis provided.')}

## Cluster Recommendation

{result.get('cluster_recommendation', 'No recommendation.')}

## Changes Applied

{change_log}

## Priority Changes Requested

{json.dumps(result.get('priority_changes', []), indent=2)}

## New Topics Suggested

{json.dumps(result.get('new_topics', []), indent=2)}
"""
    report_path.write_text(report)
    return report_path


# -- Git operations ----------------------------------------------------------

def commit_and_push(report_path: Path) -> None:
    subprocess.run(
        ["git", "config", "user.name", "pdfmux-bot"],
        check=True, cwd=REPO_ROOT,
    )
    subprocess.run(
        ["git", "config", "user.email", "bot@pdfmux.com"],
        check=True, cwd=REPO_ROOT,
    )
    subprocess.run(
        ["git", "add", str(QUEUE_FILE), str(report_path)],
        check=True, cwd=REPO_ROOT,
    )
    subprocess.run(
        ["git", "commit", "-m", f"blog: weekly keyword analysis ({report_path.stem})"],
        check=True, cwd=REPO_ROOT,
    )
    subprocess.run(["git", "push"], check=True, cwd=REPO_ROOT)


# -- Main --------------------------------------------------------------------

def main() -> None:
    print("Loading data...")
    queue = load_queue()
    published_titles = get_published_titles()
    events = load_events_calendar()
    upcoming_events = get_upcoming_events(events)
    competitors = load_competitors()
    issues = get_github_issues()

    queued_count = sum(1 for e in queue if e["status"] == "queued")
    published_count = sum(1 for e in queue if e["status"] == "published")
    print(f"  Queue: {queued_count} queued, {published_count} published")
    print(f"  Upcoming events: {len(upcoming_events)}")

    # Build prompt and run analysis
    print("Running keyword analysis via Claude API...")
    prompt = build_analysis_prompt(queue, published_titles, events, upcoming_events, issues, competitors)
    result = run_analysis(prompt)

    # Apply changes
    print("Applying changes...")
    queue, change_log = apply_changes(queue, result)
    save_queue(queue)

    # Write report
    report_path = write_report(result, change_log)
    print(f"Report: {report_path}")
    print(f"Changes:\n{change_log}")

    # Commit and push
    print("Committing...")
    commit_and_push(report_path)
    print("Done.")


if __name__ == "__main__":
    main()
