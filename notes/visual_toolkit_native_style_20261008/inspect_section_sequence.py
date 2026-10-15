"""Read-only, same-scale inspection of recorded whole-section TIFFs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
bound = json.loads((HERE / 'section_mip_diagnostic_provenance.json').read_text())
point = bound['soma_display_xy']
small, records = [], []
for entry in bound['source_sections']:
    path = Path(entry['path'])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry['sha256']:
        raise RuntimeError('Source section changed: ' + str(path))
    with tifffile.TiffFile(path) as tf:
        array = tf.asarray()
        tags = {key: str(tf.pages[0].tags[key].value)[:500]
                for key in ('ImageDescription', 'XResolution', 'YResolution', 'ResolutionUnit')
                if key in tf.pages[0].tags}
    if array.ndim != 2:
        raise ValueError('Expected one 2D section')
    shape = array.shape
    pad = np.pad(array, ((0, (-shape[0]) % 16), (0, (-shape[1]) % 16)))
    pooled = pad.reshape(pad.shape[0] // 16, 16, pad.shape[1] // 16, 16).max(axis=(1, 3))
    roi = array[3200:4000, 8827:9627]
    records.append(dict(entry, shape_yx=list(shape), dtype=str(array.dtype), tags=tags,
                        crop_nonzero_fraction=float(np.count_nonzero(roi)/roi.size),
                        crop_mean=float(roi.mean()), crop_max=int(roi.max())))
    small.append(pooled)
    del array, pad, roi
    print('Read section', entry['z_index'], flush=True)
stack = np.stack(small)
vmax = float(np.percentile(stack[stack > 0], 99.5))
fig, axes = plt.subplots(2, 5, figsize=(20, 9), facecolor='white')
for ax, plane, rec in zip(axes.ravel(), small, records):
    ax.imshow(np.sqrt(np.clip(plane.astype(float) / vmax, 0, 1)), cmap='gray', vmin=0, vmax=1)
    ax.plot(*point, '+', color='cyan', markersize=8)
    ax.set_title(str(rec['z_index']) + '\nROI nonzero ' + format(100 * rec['crop_nonzero_fraction'], '.2f') + '%')
    ax.set_axis_off()
fig.suptitle('252384 / 003: ten individual source sections; unchanged soma point\nShared display scale; 16x16 max pooling for QC only. No tissue/registration acceptance.', fontsize=14)
fig.tight_layout(rect=(0, 0, 1, .92))
png = HERE / 'sample-252384_neuron-003_desc-individualSections_alignmentQC.png'
fig.savefig(png, dpi=140)
plt.close(fig)
report = {'status':'rendered_pending_actual_inspection', 'sections':records,
          'shared_display_vmax':vmax, 'display_soma_xy':point,
          'adjacent_thumbnail_pearson':[float(np.corrcoef(small[i].ravel(), small[i+1].ravel())[0,1]) for i in range(9)],
          'figure':str(png), 'figure_sha256':hashlib.sha256(png.read_bytes()).hexdigest(),
          'source_modified':False, 'coordinates_changed':False, 'anatomical_acceptance':False}
(HERE / 'individual_section_readback.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'figure':str(png), 'section_count':len(records)}))
