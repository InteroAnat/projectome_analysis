# Cached atlas soma location audit

Checked 8746 exact live neuron identities; {'missing_unassessed': 6834, 'own_atlas_root_checked': 1606, 'source_hash_collision_unresolved': 306}. All 2480 eligible local atlas SWC files were freshly hashed and their roots read.

SWC XYZ/250 and portal physical-index um/250 give NMT voxel indices. These values are not affine world millimetres. ARM level 6 and official LR masks share the recorded NMT v2.1 symmetric grid. Missing SWCs remain unassessed; portal-coordinate lookups are a separate preliminary channel.

Source labels, candidate screens, original manual rows and hash-bound reviewed assignments are separate columns. 112 retains reviewed R-IDM; 114 remains provisional R-IDM. The working registration chain is raw fMOST brain and neurons warped to NMT. Portal-supplied label-generation and transform/version receipts remain unverified. Disagreements are descriptive findings, not proof of a portal bug or accepted relabeling.

No raw native SWCs, mirrored FNT copies, downloads or canonical edits were used. Native transform and landmark acceptance remain pending.
