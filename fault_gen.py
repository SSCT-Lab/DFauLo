import json
import os
import random

name2isfault = dict()
def gen_randomlabel_noise():
    to_noisy_data = []
    total_num = 0
    for class_name in classes:
        class_path = data_path + class_name
        data_path_list = os.listdir(class_path)
        total_num += len(data_path_list)
        sampled_data_path = random.sample(
            data_path_list, int(len(data_path_list)*falut_ratio))
                
        for sample_path in sampled_data_path:
            to_noisy_data.append((os.path.join(class_path,sample_path), class_name))
        
        for unsample_path in data_path_list:
            if unsample_path not in sampled_data_path:
                name2isfault[os.path.join(class_path,unsample_path)] = False
    
        
    class_name = list(classes.keys())
    print(len(to_noisy_data), total_num, len(to_noisy_data)/total_num)
    
    for to_noisy in to_noisy_data:
        org_class_name = to_noisy[1]
        org_data_path = to_noisy[0]
        new_class_name = random.choice(class_name)
        while new_class_name == org_class_name:
            new_class_name = random.choice(class_name)
        assert new_class_name != org_class_name
        new_data_dir_path = os.path.join(data_path, new_class_name)
        os.system(f"mv {org_data_path} {new_data_dir_path}")
        new_data_path = os.path.join(new_data_dir_path, org_data_path.split("/")[-1])
        name2isfault[new_data_path] = True
    # save json
    with open(f"{data_path}name2isfault.json", "w") as f:
        json.dump(name2isfault, f, indent=4)
        


class_path = './dataset/resisc45_classes.json'
with open(class_path, 'r') as f:
    classes = json.load(f)

noise_type = "RandomLabelNoise"
data_path = f"./dataset/{noise_type}/RESISC45/train/"

falut_ratio = 0.05
random.seed(1216)

print(classes)

gen_randomlabel_noise()
