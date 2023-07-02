import json
import os
import pickle
from os.path import isfile

import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from torchtext.data.utils import get_tokenizer


def save_as_json(data, save_path):
    data_json = json.dumps(data, indent=4)
    with open(save_path, 'w') as file:
        file.write(data_json)


def load_json(load_path):
    with open(load_path, 'r') as f:
        data = json.load(f)
    return data


class dataset(data.Dataset):
    def __init__(self, root, classes_path, transform=None, image_size=None, image_set='train', specific_label=None,
                 ignore_list=[], data_s=None, baseline=None):

        self.root = root
        self.classes_path = classes_path
        self.transform = transform
        self.image_size = image_size
        self.image_set = image_set
        self.image_root = os.path.join(self.root, self.image_set)

        if self.root.split('/')[-1] == 'MTFL':
            self.datalist, self.images, self.lms = self.getMTFL()
            delIndexList = []
            for i in range(len(self.datalist)):
                if self.datalist[i] in ignore_list:
                    delIndexList.append(i)

            for i in reversed(delIndexList):
                self.datalist = np.delete(self.datalist, i)
                self.images = np.delete(self.images, i, axis=0)
                self.lms = np.delete(self.lms, i, axis=0)

            self.images.reshape(-1, 1, 40, 40)
            self.lms.reshape(-1, 10)

            print("{} data load, {} data delete".format(len(self.datalist), len(ignore_list)))


        else:
            assert data_s is not None, 'data_s must be specified'
            assert os.path.exists(self.image_root), "dataset root path does not exist!"
            assert os.path.exists(self.classes_path), "dataset classes path does not exist!"
            # read classes(json file)
            with open(self.classes_path, 'r') as f:
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
            self.vocab = None
            if os.path.exists("./dataset/vocab.pkl"):
                self.vocab = pickle.load(open("./dataset/vocab.pkl", "rb"))
                self.tokenizer = get_tokenizer('basic_english')
            print("{} data load, {} data delete".format(len(self.datalist), len(ignore_list)))

    def getMTFL(self):
        self.root = self.root + '/' + self.image_set
        datalmPath = self.root  + '/' + self.image_set + 'ing.txt'
        annotationPath = self.root + '/annotation.txt'
        if self.image_set == 'train':
            i, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, g, s, gl, p, _ = np.genfromtxt(
                datalmPath,
                delimiter=" ",
                unpack=True)
        else:
            i, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, g, s, gl, p = np.genfromtxt(
                datalmPath,
                delimiter=" ",
                unpack=True)
        i = np.genfromtxt(datalmPath, delimiter=" ", usecols=0, dtype=str, unpack=True)

        bb1, bb2, bb3, bb4, bProb = np.genfromtxt(annotationPath, delimiter=" ", unpack=True)
        # i = [k.replace('\\', '/') for k in i]

        # Converting Annotation according to resized images
        ratio_x = 40 / (bb3 - bb1)
        ratio_y = 40 / (bb4 - bb2)
        l1, l2, l3, l4, l5, l6, l7, l8, l9, l10 = (l1 - bb1) * ratio_x, (l2 - bb1) * ratio_x, (l3 - bb1) * ratio_x, (
                l4 - bb1) * ratio_x, (l5 - bb1) * ratio_x, (l6 - bb2) * ratio_y, (l7 - bb2) * ratio_y, (
                                                          l8 - bb2) * ratio_y, (l9 - bb2) * ratio_y, (
                                                          l10 - bb2) * ratio_y
        print(self.root)
        print(i[0])
        onlyfiles = [f for f in i if isfile(os.path.join(self.root, f))]

        File_length = len(onlyfiles)
        print(File_length)
        images = list()
        indexes = list()
        imagenames = []
        for n in range(0, File_length):

            try:
                # temp = cv2.resize(cv2.imread( join(mypath, onlyfiles[n]),0),(40,40))

                temp = cv2.imread(os.path.join(self.root, onlyfiles[n]))

                temp.astype(np.uint8)
                gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
                crop_img = gray[int(bb2[n]):int(bb4[n]), int(bb1[n]):int(bb3[n])]
                if (crop_img.shape[0] < 40 or crop_img.shape[1] < 40):
                    indexes.append(n)
                    continue
                resized = cv2.resize(crop_img, (40, 40), interpolation=cv2.INTER_AREA)
                resized = resized.reshape(-1, 40, 40)
                # resized = np.expand_dims(resized,axis=2)
                imagenames.append(onlyfiles[n])
                images.append(resized)
            except:
                indexes.append(n)

        images = np.array(images)
        print(images.shape)
        for index in reversed(indexes):
            # print (index)
            i = np.delete(i, index)
            l1 = np.delete(l1, index)
            l2 = np.delete(l2, index)
            l3 = np.delete(l3, index)
            l4 = np.delete(l4, index)
            l5 = np.delete(l5, index)
            l6 = np.delete(l6, index)
            l7 = np.delete(l7, index)
            l8 = np.delete(l8, index)
            l9 = np.delete(l9, index)
            l10 = np.delete(l10, index)
            g = np.delete(g, index)
            s = np.delete(s, index)
            gl = np.delete(gl, index)
            p = np.delete(p, index)
        # images = images.reshape(10000,-1,40,40)
        print(len(l1))
        File_length = len(images)
        print(File_length)
        l1 = np.transpose(np.reshape(l1, (-1, File_length)))
        l2 = np.transpose(np.reshape(l2, (-1, File_length)))
        l3 = np.transpose(np.reshape(l3, (-1, File_length)))
        l4 = np.transpose(np.reshape(l4, (-1, File_length)))
        l5 = np.transpose(np.reshape(l5, (-1, File_length)))
        l6 = np.transpose(np.reshape(l6, (-1, File_length)))
        l7 = np.transpose(np.reshape(l7, (-1, File_length)))
        l8 = np.transpose(np.reshape(l8, (-1, File_length)))
        l9 = np.transpose(np.reshape(l9, (-1, File_length)))
        l10 = np.transpose(np.reshape(l10, (-1, File_length)))
        g = np.transpose(np.reshape(g, (-1, File_length)))
        s = np.transpose(np.reshape(s, (-1, File_length)))
        gl = np.transpose(np.reshape(gl, (-1, File_length)))
        p = np.transpose(np.reshape(p, (-1, File_length)))

        l = np.concatenate([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10], axis=1)

        gender = list()
        smile = list()
        glass = list()
        gender = list()
        pose = list()
        gender = list()
        for n in range(0, File_length):
            if g[n] == 1:
                gender.append(1)
            else:
                gender.append(0)

            if s[n] == 1:
                smile.append(1)
            else:
                smile.append(0)

            if gl[n] == 1:
                glass.append(1)
            else:
                glass.append(0)

            if p[n] == 1:
                pose.append(0)
            elif p[n] == 2:
                pose.append(1)
            elif p[n] == 3:
                pose.append(2)
            elif p[n] == 4:
                pose.append(3)
            else:
                pose.append(4)
        # result = np.concatenate([np.array(l),np.array(gender),np.array(smile),np.array(glass),np.array(pose)],axis=1)
        # print(resuz.shape)
        # print(indexes)
        imagenames = np.array(imagenames)
        assert len(images) == len(l) and len(imagenames) == len(l), "length of images and labels are not equal"
        return imagenames, images, l

    def __getitem__(self, index):
        if self.root.split('/')[-2] == 'MTFL':
            img_path = self.datalist[index]
            img = self.images[index]
            label = self.lms[index]
            return img, label, img_path

        img_path, label = self.datalist[index]
        # if txt
        if img_path[-4:] == '.txt':

            img_path = os.path.join(self.image_root, img_path)
            # read all txt in .txt as str
            with open(img_path, 'r', encoding='utf-8') as f:
                img = f.read()
            img = torch.tensor(self.vocab.transform(sentence=self.tokenizer(img), max_len=100), dtype=torch.long)

        else:
            img_path = os.path.join(self.image_root, img_path)

            img = Image.open(img_path)
            img = img.convert('RGB')

            if self.image_size is not None:
                img = img.resize(self.image_size[:2])
            if self.transform is not None:
                img = self.transform(img)

        return img, label, self.datalist[index][0]

    def __len__(self):
        return len(self.datalist)


class dataset_NCNV(data.Dataset):
    def __init__(self, root, classes_path, transform=None, image_size=None, image_set='train', specific_label=None,
                 ignore_list=[], data_s=None, mode=None, pred=None, probability=None):
        assert data_s is not None, 'data_s must be specified'
        self.root = root
        self.classes_path = classes_path
        self.transform = transform
        self.image_size = image_size
        self.image_set = image_set
        self.image_root = os.path.join(self.root, self.image_set)
        assert os.path.exists(self.image_root), "dataset root path does not exist!"
        assert os.path.exists(self.classes_path), "dataset classes path does not exist!"
        # read classes(json file)
        with open(self.classes_path, 'r') as f:
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
        self.vocab = None
        if os.path.exists("./dataset/vocab.pkl"):
            self.vocab = pickle.load(open("./dataset/vocab.pkl", "rb"))
            self.tokenizer = get_tokenizer('basic_english')
        print("{} data load, {} data delete".format(len(self.datalist), len(ignore_list)))

        self.mode = mode
        if self.mode == "labeled":
            pred_idx = pred.nonzero()[0]
            self.probability = [probability[i] for i in pred_idx]
        elif self.mode == "unlabeled":
            pred_idx = (1 - pred).nonzero()[0]
            self.unlabeled_probability = [probability[i] for i in pred_idx]

        self.datalist = [self.datalist[i] for i in pred_idx]

    def __getitem__(self, index):

        img_path, label = self.datalist[index]

        if img_path[-4:] == '.txt':
            img_path = os.path.join(self.image_root, img_path)
            # read all txt in .txt as str
            with open(img_path, 'r', encoding='utf-8') as f:
                img = f.read()
            img = torch.tensor(self.vocab.transform(sentence=self.tokenizer(img), max_len=100), dtype=torch.long)
        else:
            img_path = os.path.join(self.image_root, img_path)
            img = Image.open(img_path)
            img = img.convert('RGB')

        if self.image_size is not None:
            img = img.resize(self.image_size[:2])

        if self.mode == 'labeled':
            prob = self.probability[index]
            if self.transform == None:
                img11 = img[:]
                img12 = img[:]
                img2 = img[:]
            else:
                img11 = self.transform(img)
                img12 = self.transform(img)
                img2 = self.transform(img)
            return img11, img12, img2, label, -1, prob, index
        elif self.mode == 'unlabeled':
            prob = self.unlabeled_probability[index]

            if self.transform == None:
                img11 = img[:]
                img12 = img[:]
                img2 = img[:]
            else:
                img11 = self.transform(img)
                img12 = self.transform(img)
                img2 = self.transform(img)
            return img11, img12, img2, label, -1, prob, index
        else:
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
