

import random
import os


falut_ratio = 0.05

random.seed(1216)

data_path = "./AudioClassification-Pytorch/dataset/MAD_train_list_features.txt"
org_data_path = "./AudioClassification-Pytorch/dataset/MAD_train_list.txt"

# Random Lable Noise
rln_data_path = "./AudioClassification-Pytorch/dataset/MAD_train_rln/train/"
org_rln_data_path = "./AudioClassification-Pytorch/dataset/MAD_train_rln_list.txt"

classes = {
    "0": 0,
    "1": 1,
    "2":2,
    "3": 3,
    "4": 4,
    "5":5,
    "6":6,
}
for class_name in classes.keys():
    if not os.path.exists(rln_data_path+class_name):
        os.makedirs(rln_data_path+class_name)
        
classesid2name = {v:k for k,v in classes.items()}

with open(data_path, 'r') as f:
    lines = f.readlines()

with open(org_data_path, 'r') as f:
    org_lines = f.readlines()

idx_list = [i for i in range(len(lines))]
random.shuffle(idx_list)
idx_list = idx_list[:int(len(lines)*falut_ratio)]
name2isfault = {}
fault_cnt = 0
for i, line in enumerate(lines):
    source_path = line.split("\t")[0]
    org_source_path = org_lines[i].split("\t")[0]
    category_id = eval(line.split("\t")[1])
    new_category_id = category_id
    is_fault = False
    if i in idx_list:
        while new_category_id == category_id:
            new_category_id = random.randint(0, 6)
        assert new_category_id != category_id
        fault_cnt += 1
        is_fault = True
    new_path = f"{rln_data_path}{classesid2name[new_category_id]}/"
    new_source_name = source_path.split("/")[-1]
    os.system(f"cp ./AudioClassification-Pytorch/{source_path} {new_path}")
    name2isfault[new_path+new_source_name] = is_fault
    with open(org_rln_data_path, "a") as f:
        f.write(f"{org_source_path}\t{new_category_id}\n")
    

print("Random Lable Noise Done! Total Fault: ", fault_cnt)

# save json

import json

with open(f"{rln_data_path}name2isfault.json", "w") as f:
    json.dump(name2isfault, f, indent=4)




# print("Random Lable Noise Done! Total Fault: ", fault_cnt)

    
    
