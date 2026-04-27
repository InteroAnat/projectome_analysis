"""
Push notes/LR_insula_analysis_review.md to Get笔记 KB
'cursor_journal_projectome_analysis' as a single note.

If a note with the same title already exists in the KB, this saves a NEW
versioned note (since /note/save does not support edit). Use the agent
journal sync_to_getnote.py for proper update behavior on regular journal
files.
"""
from __future__ import annotations

import json
import os
import sys
import time

import requests

API_KEY    = os.environ.get("GETNOTE_API_KEY")
CLIENT_ID  = os.environ.get("GETNOTE_CLIENT_ID")
BASE_URL   = "https://openapi.biji.com"

PROJECT_ROOT = r"D:\projectome_analysis"
DOC_PATH = os.path.join(PROJECT_ROOT, "notes", "LR_insula_analysis_review.md")
KB_NAME = "cursor_journal_projectome_analysis"
NOTE_TITLE = "[other] LR_insula_analysis_review"


def headers():
    return {
        "Authorization": API_KEY,
        "X-Client-ID": CLIENT_ID,
        "Content-Type": "application/json",
    }


def get_or_create_kb(kb_name: str) -> str:
    """Return topic_id of the KB; create it if missing. Iterates pages."""
    print(f"[1] Listing knowledge bases to find '{kb_name}'...")
    page = 1
    found = None
    while True:
        r = requests.get(
            f"{BASE_URL}/open/api/v1/resource/knowledge/list",
            params={"page": page},
            headers=headers(), timeout=30
        )
        r.raise_for_status()
        data = r.json().get("data", {})
        for t in data.get("topics", []):
            if t.get("name") == kb_name:
                found = t
                break
        if found or not data.get("has_more"):
            break
        page += 1
    if found:
        print(f"  found KB '{kb_name}' -> topic_id={found.get('topic_id')}")
        return str(found["topic_id"])

    print(f"  KB '{kb_name}' not found, creating...")
    r = requests.post(
        f"{BASE_URL}/open/api/v1/resource/knowledge/create",
        json={"name": kb_name,
              "description": "Cursor Agent journal for projectome_analysis"},
        headers=headers(), timeout=30
    )
    r.raise_for_status()
    d = r.json().get("data", {})
    tid = str(d.get("topic_id", ""))
    print(f"  created -> topic_id={tid}")
    return tid


def update_note(note_id: str, title: str, content: str) -> bool:
    """Update existing note (replaces content)."""
    print(f"[2u] Updating existing note {note_id}...")
    r = requests.post(
        f"{BASE_URL}/open/api/v1/resource/note/update",
        json={"note_id": int(note_id), "title": title, "content": content},
        headers=headers(), timeout=60,
    )
    print(f"  status={r.status_code}")
    payload = r.json() if r.status_code == 200 else None
    if payload:
        print(f"  payload: {json.dumps(payload, indent=2, ensure_ascii=False)[:500]}")
    if r.status_code == 200 and payload and payload.get("success"):
        return True
    print(f"  body: {r.text[:500]}")
    return False


def save_note(title: str, content: str) -> str:
    """Save plain_text note; returns note_id (string)."""
    print(f"[2] Saving note '{title}' (plain_text)...")
    r = requests.post(
        f"{BASE_URL}/open/api/v1/resource/note/save",
        json={
            "title": title,
            "content": content,
            "note_type": "plain_text"
        },
        headers=headers(), timeout=60
    )
    print(f"  status={r.status_code}")
    if r.status_code != 200:
        print(f"  body={r.text[:500]}")
        r.raise_for_status()
    payload = r.json()
    print(f"  payload preview: {json.dumps(payload, indent=2, ensure_ascii=False)[:600]}")
    data = payload.get("data") or {}
    note_id = data.get("note_id") if isinstance(data, dict) else None
    if note_id is None:
        print(f"  no note_id in response. Full payload:\n{json.dumps(payload, indent=2, ensure_ascii=False)}")
        return ""
    note_id = str(note_id)
    print(f"  saved note_id={note_id}")
    return note_id


def add_to_kb(topic_id: str, note_id: str) -> bool:
    print(f"[3] Adding note {note_id} to KB {topic_id}...")
    r = requests.post(
        f"{BASE_URL}/open/api/v1/resource/knowledge/note/batch-add",
        json={"topic_id": topic_id, "note_ids": [note_id]},
        headers=headers(), timeout=30
    )
    print(f"  status={r.status_code}")
    if r.status_code != 200:
        print(f"  body={r.text[:500]}")
        return False
    print(f"  body={r.text[:200]}")
    return True


def main():
    if not API_KEY or not CLIENT_ID:
        print("[ERROR] GETNOTE_API_KEY / GETNOTE_CLIENT_ID not set")
        return 1
    if not os.path.exists(DOC_PATH):
        print(f"[ERROR] doc not found: {DOC_PATH}")
        return 2

    with open(DOC_PATH, "r", encoding="utf-8") as f:
        body = f.read()

    header_blockquote = (
        f"> 记录类型: [other] 项目分析报告 / project analysis report  \n"
        f"> 本地文件: notes/LR_insula_analysis_review.md  \n"
        f"> 知识库:  {KB_NAME}  \n"
        f"> 同步时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    content = header_blockquote + body

    print(f"Doc length: {len(body):,} chars")
    print(f"Content length (with header): {len(content):,} chars\n")

    topic_id = get_or_create_kb(KB_NAME)
    if not topic_id:
        print("[ERROR] failed to obtain topic_id")
        return 3

    # Try update on the previously-saved note first; fall back to new save
    EXISTING_NOTE_ID = "1908331896425871976"
    if update_note(EXISTING_NOTE_ID, NOTE_TITLE, content):
        print(f"\n[done] updated note_id={EXISTING_NOTE_ID} in KB {KB_NAME}")
        return 0

    print("[update failed -> trying fresh save]")
    note_id = save_note(NOTE_TITLE, content)
    if not note_id:
        print("[ERROR] failed to save note")
        return 4

    ok = add_to_kb(topic_id, note_id)
    print("\n" + ("=" * 50))
    if ok:
        print(f"[done] note_id={note_id}  in KB {topic_id} ({KB_NAME})")
    else:
        print(f"[partial] saved note_id={note_id} but batch-add failed")
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
