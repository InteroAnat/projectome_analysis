"""Persist actual visual observations after all seventeen pages were inspected."""
import json
import hashlib
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest_path = OUT / "native_contact_sheet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = []
    for page in manifest["pages"]:
        if digest(ROOT / page["path"]) != page["sha256"]:
            raise RuntimeError("Inspected contact sheet changed")
        for entry in page["sources"]:
            if digest(ROOT / entry["locator_path"]) != entry["locator_sha256"]:
                raise RuntimeError("Original locator changed")
        pages.append({"path": page["path"], "sha256": page["sha256"], "sample": page["sample"],
            "page": page["page"], "locators": len(page["sources"]), "actual_visual_inspection": True,
            "observation": "Marked native atlas-L crops occur on viewer-right and atlas-R crops on viewer-left; no obvious relative-side contradiction at contact-sheet scale"})
    report = {"status": "actual_visual_inspection_recorded", "recorded_at": datetime.now().astimezone().isoformat(),
        "manifest_sha256": digest(manifest_path), "inspected_pages": len(pages),
        "inspected_available_locators": sum(p["locators"] for p in pages), "pages": pages,
        "flags": [{"UID": "252384|003.swc", "observation": "Marked crop is in/adjacent to a large blank rectangular section area",
                   "requires": "Local source/tissue review; hemisphere position does not establish visible tissue at the soma"}],
        "scope": "Visual review of full-section crop locators; this does not inspect every soma or trace at high resolution",
        "limits": "Atlas labels are comparison annotations; absolute native orientation, transform accuracy and cytoarchitecture remain unaccepted"}
    (OUT / "native_contact_sheet_visual_readback.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "inspected_pages", "inspected_available_locators")}))

if __name__ == "__main__":
    main()
