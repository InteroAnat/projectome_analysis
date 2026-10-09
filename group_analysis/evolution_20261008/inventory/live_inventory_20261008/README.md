# All-macaque insula inventory

For animal-level information, use the [legacy-overview-backed per-monkey table and readable overview](../legacy_overview_20261009/README.md). It preserves source claims, independently checks current counts and exposes discrepancies without replacing this complete sample/channel ledger.

Generated from saved evidence: 49 exact sample IDs; 8746 exact sample/neuron identities.

Full per-neuron metadata exists for 47 samples. 0 samples have historical aggregate counts only. Blank CSV counts mean missing, never zero.

Injection claims, portal injection tags, reconstruction-list counts and candidate soma counts are separate evidence fields. Channel variants remain separate sample IDs.

Priority ranks sort actual atlas INS counts, then observed folded-box candidate lower bounds; channel variants are listed separately at the end. Incomplete potential totals are blank and retain explicit observed lower bounds. Separate G flags and INS-or-G-or-box counts preserve G without anatomical promotion. Read evidence dates and missing-count status before using a rank.

`neuron_inventory.csv` records every saved portal neuron plus visual-manifest-only identities, source hashes, labels, exclusion flags and candidate screen. `sample_inventory.csv` covers all catalog/tracker identities. `provenance.json` records sources and issues.

- All outputs are candidates only; no anatomical, layer, laterality, injection-site or VEN acceptance.
- Reconstruction-list entries are not proof of complete reconstruction; metadata tracing_cell_number is a separate field.
- Missing or failed per-neuron/soma endpoints remain unassessed; aggregate-only INS counts use historical vocabulary.
- CH1 slice evidence is dated 2026-10-02. Any later shallow folder observation has its own date and does not validate slices or PI/cytoarchitecture.
- Excluded identities remain visible and cannot enter potential counts; exact sample/channel identity is preserved.

Rerun: `python -B group_analysis/scripts/audit_insula_inventory.py`. Use `--snapshot-dir` for an additional persisted manifest. The optional `--capture-live --max-requests N` records bounded metadata GETs only under the new output folder; it never downloads images or SWCs.
