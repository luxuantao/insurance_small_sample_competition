import json
from collections import defaultdict, Counter


with open('./results/onnx_predict/ensemble_answer.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


d = defaultdict(Counter)
for k, v in data.items():
    if 'ensemble' in k:
        idx = k.find('_ensemble')
        ID = k[:idx]
        if isinstance(v, list):
            d[ID][tuple(v)] += 1
        else:
            d[ID][v] += 1
    else:
        if isinstance(v, list):
            d[ID][tuple(v)] += 1
        else:
            d[k][v] += 1

predict = {}
for ID, counter in d.items():
    predict[ID] = counter.most_common(1)[0][0]

with open('answer.json', 'w', encoding='utf-8') as f:
    json.dump(predict, f, ensure_ascii=False, indent=4)
