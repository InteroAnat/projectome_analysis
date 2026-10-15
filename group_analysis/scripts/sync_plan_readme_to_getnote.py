#!/usr/bin/env python3
"""Push plan v2 + combined_primary_v2 README to Get笔记 as plain_text, then batch-add to project KB."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import requests

BASE = "https://openapi.biji.com/open/api/v1/resource"
PROJECT = Path(__file__).resolve().parents[2]
PLAN = PROJECT / "notes/whole_insula_lr_continuation_plan_v2.md"
README = (
    PROJECT
    / "group_analysis/R_analysis/outputs/combined_primary_v2/README.md"
)
KB_NAME_HINTS = ("cursor_journal_projectome", "projectome_analysis", "cursor_journal")


def save_note(title: str, content: str, tags: list[str]) -> str:
    key = os.environ.get("GETNOTE_API_KEY", "").strip()
    cid = os.environ.get("GETNOTE_CLIENT_ID", "").strip()
    if not key or not cid:
        print("Missing GETNOTE_API_KEY or GETNOTE_CLIENT_ID in environment.", file=sys.stderr)
        sys.exit(2)
    r = requests.post(
        f"{BASE}/note/save",
        headers={
            "Authorization": key,
            "X-Client-ID": cid,
            "Content-Type": "application/json",
        },
        json={
            "title": title,
            "content": content,
            "note_type": "plain_text",
            "tags": tags,
            "parent_id": 0,
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        print(json.dumps(data, ensure_ascii=False, indent=2), file=sys.stderr)
        sys.exit(1)
    note_id = str(data.get("data", {}).get("note_id", ""))
    if not note_id:
        print(json.dumps(data, ensure_ascii=False, indent=2), file=sys.stderr)
        sys.exit(1)
    return note_id


def find_topic_id() -> str | None:
    key = os.environ["GETNOTE_API_KEY"].strip()
    cid = os.environ["GETNOTE_CLIENT_ID"].strip()
    page = 1
    candidates: list[tuple[str, str]] = []
    while page <= 20:
        r = requests.get(
            f"{BASE}/knowledge/list",
            headers={"Authorization": key, "X-Client-ID": cid},
            params={"page": page},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        if not data.get("success"):
            break
        topics = data.get("data", {}).get("topics", [])
        for t in topics:
            tid = str(t.get("topic_id", ""))
            name = (t.get("name") or "").lower()
            candidates.append((tid, t.get("name") or ""))
            for hint in KB_NAME_HINTS:
                if hint.lower() in name:
                    return tid
        if not data.get("data", {}).get("has_more"):
            break
        page += 1
    print("No KB matched hints; available:", file=sys.stderr)
    for tid, name in candidates[:30]:
        print(f"  {name!r} topic_id={tid}", file=sys.stderr)
    return None


def batch_add(topic_id: str, note_ids: list[str]) -> None:
    key = os.environ["GETNOTE_API_KEY"].strip()
    cid = os.environ["GETNOTE_CLIENT_ID"].strip()
    r = requests.post(
        f"{BASE}/knowledge/note/batch-add",
        headers={
            "Authorization": key,
            "X-Client-ID": cid,
            "Content-Type": "application/json",
        },
        json={"topic_id": topic_id, "note_ids": note_ids},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        print(json.dumps(data, ensure_ascii=False, indent=2), file=sys.stderr)
        sys.exit(1)
    print("batch-add:", json.dumps(data.get("data"), ensure_ascii=False))


def main() -> None:
    if not PLAN.is_file():
        print("Missing plan:", PLAN, file=sys.stderr)
        sys.exit(1)
    if not README.is_file():
        print("Missing README:", README, file=sys.stderr)
        sys.exit(1)
    plan_body = PLAN.read_text(encoding="utf-8")
    readme_body = README.read_text(encoding="utf-8")
    plan_content = (
        f"**Local source:** `{PLAN.as_posix()}`\n\n---\n\n" + plan_body
    )
    readme_content = (
        f"**Local source:** `{README.as_posix()}`\n\n---\n\n" + readme_body
    )

    tags = ["projectome_analysis", "combined_primary_v2", "plan_v2"]

    print("Saving plan note…")
    id1 = save_note(
        "[plan] whole_insula_lr_continuation_plan_v2",
        plan_content,
        tags + ["LR_insula"],
    )
    print("  note_id:", id1)

    print("Waiting 65s before second save (API rate limit guidance)…")
    time.sleep(65)

    print("Saving README note…")
    id2 = save_note(
        "[readme] combined_primary_v2 README",
        readme_content,
        tags + ["README"],
    )
    print("  note_id:", id2)

    topic_id = find_topic_id()
    if topic_id:
        print("Adding to knowledge base topic_id:", topic_id)
        batch_add(topic_id, [id1, id2])
    else:
        print("Skipped KB add: create or rename a KB containing", KB_NAME_HINTS)
        print("Then batch-add note_ids manually:", id1, id2)


if __name__ == "__main__":
    main()
