import json
from collections import defaultdict

preds  = json.load(open('./guardian_test_logs/agentharm/gpt-4o-2024-08-06/preds.json'))
labels = json.load(open('./guardian_test_logs/agentharm/gpt-4o-2024-08-06/labels.json'))
meta   = json.load(open('./guardian_test_logs/agentharm/gpt-4o-2024-08-06/meta_data.json'))

def strict(x): return 0 if x == 0.0 else 1

results = defaultdict(lambda: {'tp':0,'fn':0,'fp':0,'tn':0})

for p, l, m in zip(preds, labels, meta):
    style = m['meta_sample'].get('injection_style', 'unknown')
    delay = m['meta_sample'].get('injection_delay', -1)
    pm, lm = strict(p), strict(l)

    for key in [style, f'delay_{delay}']:
        if   lm==1 and pm==1: results[key]['tp'] += 1
        elif lm==1 and pm==0: results[key]['fn'] += 1
        elif lm==0 and pm==1: results[key]['fp'] += 1
        else:                 results[key]['tn'] += 1

print(f'{"Stratum":<20} {"TP":>4} {"FN":>4} {"FP":>4} {"TN":>4}  {"Recall":>7}  {"Prec":>7}')
print('-' * 60)

for k in sorted(results):
    r = results[k]
    rec  = r['tp']/(r['tp']+r['fn']) if (r['tp']+r['fn'])>0 else 0
    prec = r['tp']/(r['tp']+r['fp']) if (r['tp']+r['fp'])>0 else 0

    print(f'{k:<20} {r["tp"]:>4} {r["fn"]:>4} {r["fp"]:>4} {r["tn"]:>4}  {rec:>7.3f}  {prec:>7.3f}')