from instructions.instruction_templates import instruction_format
from utils.utils import write2json, load_json
import random
from itertools import permutations  
from copy import deepcopy


EVAL_RATIO = 0.2
AUG_TRAIN_DATA = False
MAX_ALLOW_CAND = 3


def aug_data(data):
    new_data1 = []
    for each in data:
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
        },
        '''
        new_data1.append(each)

        # aug verbalizer
        if 'verbalizer' in each:
            origin_verbalizer = each['verbalizer']
            origin_instruction = each['instruction']
            verbalizers = each['verbalizer'].split('/')
            if len(verbalizers) > MAX_ALLOW_CAND:  # 有些候选太多的就不换了
                continue
            
            perm = permutations(verbalizers)
            for new_perm in list(perm):
                new_verbalizer = '/'.join(new_perm)
                if new_verbalizer == origin_verbalizer:
                    continue
                new_instruction = origin_instruction.replace(origin_verbalizer, new_verbalizer)
                
                new = deepcopy(each)
                new['ID'] = each['ID'] + '_aug1'
                new['verbalizer'] = new_verbalizer
                new['instruction'] = new_instruction
                
                new_data1.append(new)
    
    new_data2 = []
    for each in new_data1:
        new_data2.append(each)
        
        # text_a text_b swap
        if 'text_b' in each:
            origin_instruction = each['instruction']
            text_a = each['text_a']
            text_b = each['text_b']
            idx_a = origin_instruction.find(text_a)
            idx_b = origin_instruction.find(text_b)
            new_instruction = origin_instruction[:idx_a] + text_b + origin_instruction[idx_a+len(text_a):idx_b] + text_a + origin_instruction[idx_b+len(text_b):]
            
            new = deepcopy(each)
            new['ID'] = each['ID'] + '_aug2'
            new['instruction'] = new_instruction
            new['text_a'] = text_b
            new['text_b'] = text_a

            new_data2.append(new)
            
    # random.shuffle(new_data2)
    return new_data2


all_data = load_json("./data/train_data.json")
all_data = instruction_format(all_data)
write2json(all_data, "./data/instruction_all.json", "all data")
print(len(all_data))

EVAL_NUM = int(EVAL_RATIO * len(all_data))
random.seed(42)
random.shuffle(all_data)

train_data = all_data[:-EVAL_NUM]
if AUG_TRAIN_DATA:
    train_data = aug_data(train_data)
write2json(train_data, "./data/instruction_train.json", "train data")
print(len(train_data))

dev_data = all_data[-EVAL_NUM:]
write2json(dev_data, "./data/instruction_dev.json", "dev data")
print(len(dev_data))

test_data = load_json("./data/test_data_A.json")
test_data = instruction_format(test_data)
write2json(test_data, "./data/instruction_test.json", "test data")
print(len(test_data))

