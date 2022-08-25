import cv2
import numpy as np
from os.path import isfile, join
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile, join


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mypath, train, annotations, transform=None):

        self.images, self.landmark, self.gender, self.smile, self.glass, self.pose = self.load_dataset(mypath, train,
                                                                                                       annotations)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        landmark = self.landmark[idx]
        glass = self.glass[idx]
        smile = self.smile[idx]
        pose = self.pose[idx]
        gender = self.gender[idx]
        return image, landmark, gender, smile, glass, pose

    def load_dataset(self, mypath, train, annotation):
        # i,l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,g,s,gl,p = np.genfromtxt("/media/dongy/Windows7_OS/Users/Owner/Desktop/Life with Divine/MTFL/training.txt", delimiter=" ", unpack=True)
        #isdirty,,isVAE,isMID,isOUT
        i, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, g, s, gl, p= np.genfromtxt(join(mypath, train),#,isVAE,isMID,isOUT,isdirty,isVAE,isMID,isOUT
                                                                                                delimiter=" ",
                                                                                                unpack=True)
        i = np.genfromtxt(join(mypath, train), delimiter=" ", usecols=0, dtype=str, unpack=True)
        bb1, bb2, bb3, bb4, bProb = np.genfromtxt(join(mypath, annotation), delimiter=" ", unpack=True)
        i = [k.replace('\\', '/') for k in i]

        # Converting Annotation according to resized images
        ratio_x = 40 / (bb3 - bb1)
        ratio_y = 40 / (bb4 - bb2)
        l1, l2, l3, l4, l5, l6, l7, l8, l9, l10 = (l1 - bb1) * ratio_x, (l2 - bb1) * ratio_x, (l3 - bb1) * ratio_x, (
                l4 - bb1) * ratio_x, (l5 - bb1) * ratio_x, (l6 - bb2) * ratio_y, (l7 - bb2) * ratio_y, (
                                                          l8 - bb2) * ratio_y, (l9 - bb2) * ratio_y, (
                                                          l10 - bb2) * ratio_y

        onlyfiles = [f for f in i if isfile(join(mypath, f))]
        File_length = len(onlyfiles)
        images = list()
        indexes = list()

        for n in range(0, File_length):
            try:
                # temp = cv2.resize(cv2.imread( join(mypath, onlyfiles[n]),0),(40,40))
                temp = cv2.imread(join(mypath, onlyfiles[n]))
                temp.astype(np.uint8)
                gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
                crop_img = gray[int(bb2[n]):int(bb4[n]), int(bb1[n]):int(bb3[n])]
                if (crop_img.shape[0] < 40 or crop_img.shape[1] < 40): #or int(isVAE[n])==1):
                    indexes.append(n)
                    continue
                resized = cv2.resize(crop_img, (40, 40), interpolation=cv2.INTER_AREA)
                resized = resized.reshape(-1, 40, 40)
                # resized = np.expand_dims(resized,axis=2)
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
        print(images.shape)
        return images, l, np.array(gender), np.array(smile), np.array(glass), np.array(pose)
