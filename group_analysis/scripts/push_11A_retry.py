"""Retry 11A push after QPS rate-limit cooldown."""
import os, sys, json, time, re, requests

API_KEY = os.environ.get("GETNOTE_API_KEY")
CLIENT_ID = os.environ.get("GETNOTE_CLIENT_ID")
BASE = "https://openapi.biji.com"

DOC = r"D:\projectome_analysis\notes\LR_insula_analysis_review.md"
KB_NAME = "cursor_journal_projectome_analysis"
TITLE_11A = "[other] LR_insula_review-11A_multi_monkey_extension"


def hdrs():
    return {"Authorization": API_KEY, "X-Client-ID": CLIENT_ID,
            "Content-Type": "application/json"}


with open(DOC, "r", encoding="utf-8") as f:
    body = f.read()

m_11a = re.search(r"^## 11A\.", body, flags=re.M)
m_11b = re.search(r"^## 11B\.", body, flags=re.M)
sec_11A = body[m_11a.start():m_11b.start()]
hdr = (
    f"> 记录类型: [other] 项目分析报告 / project analysis report  \n"
    f"> 本地文件: notes/LR_insula_analysis_review.md  \n"
    f"> 知识库:  {KB_NAME}  \n"
    f"> 同步时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  \n"
    f"> 部分:    Section 11A multi-monkey extension\n\n"
)
content = hdr + sec_11A
print(f"Content length: {len(content):,}")

# Find KB
r = requests.get(f"{BASE}/open/api/v1/resource/knowledge/list",
                  params={"page": 1}, headers=hdrs(), timeout=30)
topics = r.json().get("data", {}).get("topics", [])
topic_id = next((str(t["topic_id"]) for t in topics if t.get("name") == KB_NAME), None)
print(f"topic_id={topic_id}")

# Cooldown then save
print("Sleep 30s for QPS bucket reset...")
time.sleep(30)

print("Saving 11A...")
r = requests.post(f"{BASE}/open/api/v1/resource/note/save",
                   json={"title": TITLE_11A, "content": content,
                         "note_type": "plain_text"},
                   headers=hdrs(), timeout=60)
payload = r.json() if r.status_code == 200 else None
print(f"status={r.status_code} body={r.text[:300]}")
if not (payload and payload.get("success")):
    print("[FAIL]")
    sys.exit(1)
note_id = str((payload.get("data") or {}).get("note_id", ""))
print(f"saved note_id={note_id}")

time.sleep(5)
print("Add to KB...")
r2 = requests.post(f"{BASE}/open/api/v1/resource/knowledge/note/batch-add",
                    json={"topic_id": topic_id, "note_ids": [note_id]},
                    headers=hdrs(), timeout=30)
print(f"status={r2.status_code} body={r2.text[:300]}")
print("[DONE]")
