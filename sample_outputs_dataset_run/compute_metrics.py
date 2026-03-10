import json
import glob

files = sorted(glob.glob('moiz_kashif_csc481_final/sample_outputs_dataset_run/*.json'))
if not files:
    raise RuntimeError('no json files found in sample_outputs_dataset_run')

n = len(files)
non_null = 0
plaus = 0
conflicts = 0

for f in files:
    with open(f, 'r', encoding='utf-8') as fh:
        d = json.load(fh)

    a = (d.get('raw') or {}).get('path_a_metrics') or {}
    lap = a.get('laplacian_variance', None)
    bright = a.get('brightness_mean', None)

    ok = (lap is not None) and (bright is not None) and (float(lap) > 0.0)
    if ok:
        non_null += 1

    notes = d.get('notes', []) or []
    has_conflict = any('conflict' in str(x).lower() for x in notes)
    if has_conflict:
        conflicts += 1
    else:
        plaus += 1

print('N', n)
print('classical_non_null_rate', round(non_null / n, 4))
print('semantic_plausibility_rate_no_conflict_notes', round(plaus / n, 4))
print('conflict_rate', round(conflicts / n, 4))
