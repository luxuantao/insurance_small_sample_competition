import json
import random
from tqdm import tqdm
from copy import deepcopy
from instructions.instruction_templates import instruction_format
from utils.utils import write2json, load_json


random.seed(42)
extra_num = 4


test_data = load_json("./data/test_data_B.json")
test_data = instruction_format(test_data)
write2json(test_data, "./data/instruction_test_b.json", "test data")
print(len(test_data))

with open('data/instruction_test_b.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

new_data = []
for each in tqdm(data):
    '''
        {
            "ID": "MedicalNLI_TRAIN_3",
            "text_a": "多数阿尔茨海默病病例为散发，晚发型发病年龄≥65岁，病因不明。",
            "text_b": "老年人更容易患阿尔茨海默病",
            "label": "蕴含",
            "target": "蕴含",
            "verbalizer": "蕴含/无关/矛盾",
            "instruction": "【多数阿尔茨海默病病例为散发，晚发型发病年龄≥65岁，病因不明。】和【老年人更容易患阿尔茨海默病】，以上两句话的逻辑关系是什么？蕴含/无关/矛盾",
            "data_type": "classification"
        }
    '''
    s = set()
    if 'verbalizer' in each:
        verbalizer = each['verbalizer']
        instruction = each['instruction']
        cands = verbalizer.split('/')
        if len(cands) < 3:
            new_data.append(each)
            continue
        
        max_extra = 1
        for i in range(2, len(cands) + 1):
            max_extra *= i
        max_extra -= 1  # 1是原本
        
        s.add(tuple(cands))
        i = 0
        while i < extra_num and i < max_extra:
            random.shuffle(cands)
            if tuple(cands) not in s:
                i += 1
                s.add(tuple(cands))
                new_verbalizer = '/'.join(cands)
                new_instruction = instruction.replace(verbalizer, new_verbalizer)
                
                new = deepcopy(each)
                new['ID'] = each['ID'] + f'_ensemble_{i}'
                new['verbalizer'] = new_verbalizer
                new['instruction'] = new_instruction
                
                new_data.append(new)

    new_data.append(each)

print(len(new_data))

with open('data/ensemble_instruction_test_b.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
