import json
import os

import torch.utils.data as data
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import transforms


class dataset(data.Dataset):
    def __init__(self, root, classes_path, transform=None, image_size=None, image_set='train', specific_label=None,
                 ignore_list=[], data_s=None):
        assert data_s is not None, 'data_s must be specified'
        self.root = root
        self.classes_path = classes_path
        self.transform = transform
        self.image_size = image_size
        self.image_set = image_set
        self.image_root = os.path.join(self.root, self.image_set)
        assert os.path.exists(self.image_root), "dataset root path does not exist!"
        assert os.path.exists(os.path.join(root, self.classes_path)), "dataset classes path does not exist!"
        # read classes(json file)
        with open(os.path.join(root, self.classes_path), 'r') as f:
            self.classes = json.load(f)
        class_keys = list(self.classes.keys())
        if specific_label is None:
            self.datalist = []
            for name in class_keys:
                img_list = []
                for path in data_s[name]:
                    if path not in ignore_list:
                        img_list.append(path)
                datas = zip(img_list, [self.classes[name]] * len(img_list))
                datas = list(datas)
                self.datalist.extend(datas)
        else:
            self.datalist = []
            name = specific_label
            img_list = []
            for path in data_s[name]:
                if path not in ignore_list:
                    img_list.append(path)
            datas = zip(img_list, [self.classes[name]] * len(img_list))
            datas = list(datas)
            self.datalist.extend(datas)
        print("{} data load, {} data delete".format(len(self.datalist), len(ignore_list)))

    def __getitem__(self, index):
        img_path, label = self.datalist[index]
        img_path = os.path.join(self.image_root, img_path)
        img = Image.open(img_path)
        img = img.convert('RGB')
        if self.image_size is not None:
            if self.image_size[2] == 1:
                img = img.convert('L')
            img = img.resize(self.image_size[:2])
        if self.transform is not None:
            img = self.transform(img)

        return img, label, self.datalist[index][0]

    def __len__(self):
        return len(self.datalist)


if __name__ == '__main__':
    dataset(root='../dataset/mnist', classes_path='classes.json')

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    data_loader = data.DataLoader(
        dataset(root='../dataset/mnist', classes_path='classes.json', transform=transform),
        batch_size=1, shuffle=True, num_workers=0)

    for img, label in data_loader:
        print(img.shape, label)
        # plt
        plt.imshow(img[0][0], cmap='gray')
        plt.show()
