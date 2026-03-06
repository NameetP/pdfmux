"""Data enrichment module for pdfmux blog content pipeline.

Pulls real, verifiable data that gets injected into the LLM system prompt
so every blog post has fresh numbers that generic AI content cannot have.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import requests


def get_pypi_downloads() -> str:
    """Pull recent download count from pypistats.org (no auth needed)."""
    try:
        resp = requests.get(
            "https://pypistats.org/api/packages/pdfmux/recent",
            timeout=10,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        last_day = data.get("last_day", "unknown")
        last_week = data.get("last_week", "unknown")
        last_month = data.get("last_month", "unknown")
        return (
            f"PyPI downloads: {last_day}/day, {last_week}/week, {last_month}/month"
        )
    except Exception as e:
        return f"PyPI downloads: unavailable ({e})"


def get_github_stats() -> str:
    """Pull GitHub stars, forks, open issues via gh CLI."""
    try:
        result = subprocess.run(
            ["gh", "api", "repos/NameetP/pdfmux", "--jq",
             '{stars: .stargazers_count, forks: .forks_count, '
             'issues: .open_issues_count, watchers: .subscribers_count}'],
            capture_output=True, text=True, timeout=15, check=True,
        )
        data = json.loads(result.stdout)
        return (
            f"GitHub: {data['stars']} stars, {data['forks']} forks, "
            f"{data['issues']} open issues, {data['watchers']} watchers"
        )
    except Exception as e:
        return f"GitHub stats: unavailable ({e})"


def get_github_recent_issues(limit: int = 5) -> str:
    """Pull recent GitHub issues for content ideas."""
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--repo", "NameetP/pdfmux",
             "--limit", str(limit), "--json", "title,number,labels",
             "--state", "open"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        issues = json.loads(result.stdout)
        if not issues:
            return "Recent GitHub issues: none open"
        lines = [f"  #{i['number']}: {i['title']}" for i in issues]
        return "Recent GitHub issues:\n" + "\n".join(lines)
    except Exception:
        return "Recent GitHub issues: unavailable"


def get_current_date_context() -> str:
    """Return current date for freshness signals."""
    now = datetime.now(timezone.utc)
    return f"Current date: {now.strftime('%B %d, %Y')} (use this for freshness)"


def build_enrichment_block() -> str:
    """Combine all enrichment sources into a text block for the system prompt."""
    sections = [
        "## Real Data (use these exact numbers in the post where relevant)",
        "",
        get_current_date_context(),
        get_pypi_downloads(),
        get_github_stats(),
        get_github_recent_issues(),
    ]
    return "\n".join(sections)
