"""Push the LR review to Get笔记 in 3 split notes (each under 50KB):
  Note 1 (update existing): Sections 1-11 base + 11C theoretical reframing
  Note 2 (new):             Section 11A multi-monkey extension
  Note 3 (new):             Section 11B systematic analyses
"""
from __future__ import annotations

import json
import os
import sys
import time
import re
import requests

API_KEY    = os.environ.get("GETNOTE_API_KEY")
CLIENT_ID  = os.environ.get("GETNOTE_CLIENT_ID")
BASE_URL   = "https://openapi.biji.com"

PROJECT_ROOT = r"D:\projectome_analysis"
DOC_PATH = os.path.join(PROJECT_ROOT, "notes", "LR_insula_analysis_review.md")
KB_NAME = "cursor_journal_projectome_analysis"
EXISTING_NOTE_ID = "1908331896425871976"


def headers():
    return {
        "Authorization": API_KEY,
        "X-Client-ID": CLIENT_ID,
        "Content-Type": "application/json",
    }


def make_blockquote(file_label: str, part_label: str = "") -> str:
    return (
        f"> 记录类型: [other] 项目分析报告 / project analysis report  \n"
        f"> 本地文件: {file_label}  \n"
        f"> 知识库:  {KB_NAME}  \n"
        f"> 同步时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        + (f"  \n> 部分:    {part_label}" if part_label else "")
        + "\n\n"
    )


def get_or_create_kb(kb_name: str) -> str:
    page = 1
    while True:
        r = requests.get(
            f"{BASE_URL}/open/api/v1/resource/knowledge/list",
            params={"page": page}, headers=headers(), timeout=30,
        )
        r.raise_for_status()
        data = r.json().get("data", {})
        for t in data.get("topics", []):
            if t.get("name") == kb_name:
                return str(t["topic_id"])
        if not data.get("has_more"):
            break
        page += 1
    r = requests.post(
        f"{BASE_URL}/open/api/v1/resource/knowledge/create",
        json={"name": kb_name,
              "description": "Cursor Agent journal for projectome_analysis"},
        headers=headers(), timeout=30,
    )
    r.raise_for_status()
    return str(r.json().get("data", {}).get("topic_id", ""))


def update_note(note_id: str, title: str, content: str) -> bool:
    print(f"  -> update note_id={note_id}, title={title!r}, len={len(content):,}")
    r = requests.post(
        f"{BASE_URL}/open/api/v1/resource/note/update",
        json={"note_id": int(note_id), "title": title, "content": content},
        headers=headers(), timeout=60,
    )
    if r.status_code != 200:
        print(f"    HTTP {r.status_code}: {r.text[:300]}")
        return False
    payload = r.json()
    ok = bool(payload.get("success"))
    if not ok:
        print(f"    err: {json.dumps(payload, ensure_ascii=False)[:300]}")
    return ok


def save_note(title: str, content: str) -> str:
    print(f"  -> save title={title!r}, len={len(content):,}")
    r = requests.post(
        f"{BASE_URL}/open/api/v1/resource/note/save",
        json={"title": title, "content": content, "note_type": "plain_text"},
        headers=headers(), timeout=60,
    )
    if r.status_code != 200:
        print(f"    HTTP {r.status_code}: {r.text[:300]}")
        return ""
    payload = r.json()
    if not payload.get("success"):
        print(f"    err: {json.dumps(payload, ensure_ascii=False)[:300]}")
        return ""
    note_id = (payload.get("data") or {}).get("note_id")
    return str(note_id) if note_id else ""


def add_to_kb(topic_id: str, note_id: str) -> bool:
    r = requests.post(
        f"{BASE_URL}/open/api/v1/resource/knowledge/note/batch-add",
        json={"topic_id": topic_id, "note_ids": [note_id]},
        headers=headers(), timeout=30,
    )
    return r.status_code == 200 and r.json().get("success", False)


def split_doc_at_section_11A(text: str) -> tuple[str, str, str]:
    """Returns (head, sec_11A, sec_11B_plus_11C)."""
    # Markers in current doc
    m_11a = re.search(r"^## 11A\.", text, flags=re.M)
    m_11b = re.search(r"^## 11B\.", text, flags=re.M)
    m_11c = re.search(r"^## 11C\.", text, flags=re.M)

    if m_11a is None or m_11b is None:
        raise SystemExit("could not find Section 11A or 11B headers")

    head = text[:m_11a.start()]
    sec_11A = text[m_11a.start():m_11b.start()]
    sec_11B_C = text[m_11b.start():]
    return head, sec_11A, sec_11B_C


def main():
    if not API_KEY or not CLIENT_ID:
        print("[ERROR] credentials not set")
        return 1
    if not os.path.exists(DOC_PATH):
        print(f"[ERROR] {DOC_PATH} missing")
        return 2

    with open(DOC_PATH, "r", encoding="utf-8") as f:
        body = f.read()
    print(f"Doc length: {len(body):,}")

    # Cut at major sections
    head, sec_11A, sec_11B_C = split_doc_at_section_11A(body)
    print(f"  head:        {len(head):,} chars")
    print(f"  sec 11A:     {len(sec_11A):,} chars")
    print(f"  sec 11B+11C: {len(sec_11B_C):,} chars")

    # Compose 3 notes:
    # Note 1: head (sections 1-11; ends just before 11A)
    # Note 2: 11A
    # Note 3: 11B + 11C
    note1_title = "[other] LR_insula_analysis_review"
    note2_title = "[other] LR_insula_review-11A_multi_monkey_extension"
    note3_title = "[other] LR_insula_review-11B+11C_systematic+reframing"

    note1_body = make_blockquote(
        "notes/LR_insula_analysis_review.md", "Sections 1-11 (base)"
    ) + head
    note2_body = make_blockquote(
        "notes/LR_insula_analysis_review.md", "Section 11A multi-monkey extension"
    ) + sec_11A
    note3_body = make_blockquote(
        "notes/LR_insula_analysis_review.md",
        "Section 11B (systematic L6 + intra-insula) + Section 11C (theoretical reframing)"
    ) + sec_11B_C

    print(f"\nNote payloads:")
    print(f"  note1 ({note1_title!r}): {len(note1_body):,} chars")
    print(f"  note2 ({note2_title!r}): {len(note2_body):,} chars")
    print(f"  note3 ({note3_title!r}): {len(note3_body):,} chars")

    if any(len(s) > 50_000 for s in (note1_body, note2_body, note3_body)):
        print("[WARN] some part still > 50000 chars; may exceed API limit")

    topic_id = get_or_create_kb(KB_NAME)
    print(f"\n[KB] topic_id={topic_id}")

    # Note 1: update existing
    print("\n[1] update existing main note ...")
    ok = update_note(EXISTING_NOTE_ID, note1_title, note1_body)
    if not ok:
        print("  [FAIL] update; trying fresh save")
        new_id = save_note(note1_title, note1_body)
        if new_id:
            add_to_kb(topic_id, new_id)
            print(f"  [ok] new note saved {new_id}")
        else:
            print(f"  [FAIL] could not save note1")
    else:
        print(f"  [ok] updated note_id={EXISTING_NOTE_ID}")

    # Notes 2, 3: new save + batch-add
    for title, body_text, label in (
        (note2_title, note2_body, "11A"),
        (note3_title, note3_body, "11B+11C"),
    ):
        print(f"\n[{label}] save new ...")
        nid = save_note(title, body_text)
        if not nid:
            print(f"  [FAIL] could not save {label}")
            continue
        added = add_to_kb(topic_id, nid)
        print(f"  [ok] note_id={nid}  added_to_KB={added}")
        time.sleep(2)  # rate-limit safety

    print("\n[DONE]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
