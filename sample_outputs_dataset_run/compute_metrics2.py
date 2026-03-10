import json
import glob

files = sorted(glob.glob('moiz_kashif_csc481_final/sample_outputs_dataset_run/*.json'))
if not files:
    raise RuntimeError('no json files found')

n=len(files)
low_conf=0
for f in files:
    d=json.load(open(f,'r',encoding='utf-8'))
    conf=float(d.get('final_scene_confidence',0.0) or 0.0)
    if conf < 0.60:
        low_conf += 1
print('N', n)
print('low_confidence_rate_lt_0_60', round(low_conf/n,4))
