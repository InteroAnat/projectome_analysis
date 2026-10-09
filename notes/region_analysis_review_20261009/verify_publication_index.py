"""Verify exact Git-index bytes against the scoped publication manifest."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import re
import subprocess
from prepare_publication import ACCESS_TOKEN_PATTERN

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent
GIT = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
INDEX_CREDENTIAL_PATTERN = re.compile(
    b"(?:" + ACCESS_TOKEN_PATTERN.encode("ascii") +
    rb"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)


def main():
    manifest = AUDIT / "publication_files.json"
    record = json.loads(manifest.read_text(encoding="utf-8"))
    expected = {item["path"]: item["sha256"] for items in record["groups"].values() for item in items}
    extra = {f"notes/region_analysis_review_20261009/{name}" for name in
             ("publication_files.json", "publication_scan.json", "publication_local_artifacts.json",
              "publication_paths_baseline.txt", "publication_paths_repairs.txt", "publication_paths_evidence.txt")}
    staged = set(subprocess.check_output(GIT + ["diff", "--cached", "--name-only", "-z"], cwd=ROOT).decode("utf-8").split("\0")) - {""}
    unexpected = sorted(staged - set(expected) - extra)
    errors = [f"Out-of-scope staged path: {path}" for path in unexpected]
    # The inherited credential is used only as an in-memory fingerprint.
    inherited = subprocess.check_output(GIT + ["show", "78008cbaab71bf737dc7129c72e5fe284c3ba40f:main_scripts/Visual_toolkit.py"], cwd=ROOT).decode("utf-8")
    secrets = [node.value.value for node in ast.walk(ast.parse(inherited))
               if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
               and isinstance(node.value.value, str) and len(node.value.value) >= 4
               and any(isinstance(target, ast.Name) and target.id == "SSH_PASS" for target in node.targets)]
    pattern = INDEX_CREDENTIAL_PATTERN
    checked, total, findings = {}, 0, []
    process = subprocess.Popen(GIT + ["cat-file", "--batch"], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for path in sorted(set(expected) | extra):
            process.stdin.write(f":{path}\n".encode("utf-8"))
            process.stdin.flush()
            header = process.stdout.readline().decode("utf-8").strip().split()
            if len(header) != 3 or header[1] != "blob":
                errors.append(f"Missing index blob: {path}")
                continue
            size = int(header[2])
            data = process.stdout.read(size)
            process.stdout.read(1)
            actual = hashlib.sha256(data).hexdigest()
            working = (ROOT / path).read_bytes()
            if data != working or (path in expected and actual != expected[path]):
                errors.append(f"Index/working/manifest byte mismatch: {path}")
            if pattern.search(data) or any(value.encode("utf-8") in data for value in secrets):
                findings.append({"path": path, "category": "credential_bytes", "value_logged": False})
            checked[path] = actual
            total += size
    finally:
        process.stdin.close()
        process.wait()
    errors += [f"Credential scan finding: {row['path']}" for row in findings]
    result = {"status": "passed" if not errors else "failed", "checked_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "index_blobs_sha256": checked, "checked_files": len(checked), "bytes_checked": total,
              "staged_paths": len(staged), "unexpected_staged_paths": unexpected,
              "credential_findings": findings, "credential_values_logged": False,
              "hash_preservation": "Exact source/derivative bytes retained; command-scoped core.autocrlf=false avoids rewriting hash-bound artifacts",
              "errors": errors}
    (AUDIT / "publication_index_receipt.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "checked_files", "bytes_checked", "staged_paths", "errors")}))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
