import argparse
import gc
import json
import os
import random
import sys
import time
import warnings

import numpy as np

from pyod.models.vae import VAE
from sklearn.cluster import KMeans
from torch import nn, utils
from tqdm import tqdm
import torch
from utils.dataset import dataset
from utils.models import *
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DfauLo():
    def __init__(self, args):
        self.args = args
        classes = self.load_json(self.args.class_path)
        self.class_num = len(classes.keys())
        # creat dir
        if not os.path.exists(os.path.join(self.args.dataset, 'feature/' + self.args.model_name)):
            os.makedirs(os.path.join(self.args.dataset, 'feature/' + self.args.model_name))
        if not os.path.exists(os.path.join(self.args.dataset, 'results/' + self.args.model_name)):
            os.makedirs(os.path.join(self.args.dataset, 'results/' + self.args.model_name))
        if not os.path.exists(os.path.join(self.args.dataset, 'dellist/' + self.args.model_name)):
            os.makedirs(os.path.join(self.args.dataset, 'dellist/' + self.args.model_name))
        if not os.path.exists(os.path.join(self.args.dataset, 'mutmodel/' + self.args.model_name)):
            os.makedirs(os.path.join(self.args.dataset, 'mutmodel/' + self.args.model_name))

    def run(self, data_s):
        dfaulo_time = {
            'Select Subset': -1,
            'Mutation&Extraction': -1,
            'Initialize Susp': -1,
            'Update Susp': -1,
            'all': -1
        }

        if os.path.exists(
                os.path.join(self.args.dataset, 'results/' + self.args.model_name + '/noManual_results_list.json')) and \
                os.path.exists(os.path.join(self.args.dataset,
                                            'feature/' + self.args.model_name + '/noManual_full_Feature.json')) and \
                os.path.exists(os.path.join(self.args.dataset,
                                            'results/' + self.args.model_name + '/noManual_sorted_score_list.json')) and self.args.ablation == 'None':
            print('noManual_results_list.json and noManual_full_Feature.json already exist!')
            noManual_results_list = self.load_json(
                os.path.join(self.args.dataset, 'results/' + self.args.model_name + '/noManual_results_list.json'))
            noManual_full_Feature = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/noManual_full_Feature.json'))
            noManual_sorted_score_list = self.load_json(
                os.path.join(self.args.dataset, 'results/' + self.args.model_name + '/noManual_sorted_score_list.json'))
        else:
            if self.args.ablation == 'None':
                if os.path.exists(
                        os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/full_Feature.json')):
                    Feature = self.load_json(
                        os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/full_Feature.json'))
                    print('Feature loaded')
                else:
                    Feature, Select_Subset_time, Mutation_Extraction_time = self.Feature_Summary(data_s)
                    dfaulo_time['Select Subset'] = Select_Subset_time
                    dfaulo_time['Mutation&Extraction'] = Mutation_Extraction_time
                    self.save_as_json(Feature,
                                      os.path.join(self.args.dataset,
                                                   'feature/' + self.args.model_name + '/full_Feature.json'))
            else:
                feature_type = self.args.ablation
                if os.path.exists(
                        os.path.join(self.args.dataset,
                                     'feature/' + self.args.model_name + '/full_Feature_' + feature_type + '.json')):
                    Feature = self.load_json(
                        os.path.join(self.args.dataset,
                                     'feature/' + self.args.model_name + '/full_Feature_' + feature_type + '.json'))
                    print('Feature loaded')
                else:
                    Feature, Select_Subset_time, Mutation_Extraction_time = self.Feature_Summary(data_s)
                    dfaulo_time['Select Subset'] = Select_Subset_time
                    dfaulo_time['Mutation&Extraction'] = Mutation_Extraction_time
                    self.save_as_json(Feature,
                                      os.path.join(self.args.dataset,
                                                   'feature/' + self.args.model_name + '/full_Feature_' + feature_type + '.json'))
            print('start nomanual iteration')
            noManual_results_list, noManual_full_Feature, noManual_sorted_score_list, Initialize_Susp_time = self.Iteration(
                Feature)
            print('nomanual iteration finished')
            dfaulo_time['Initialize Susp'] = Initialize_Susp_time
            if self.args.ablation == 'None':
                self.save_as_json(noManual_results_list,
                                  os.path.join(self.args.dataset,
                                               'results/' + self.args.model_name + '/noManual_results_list.json'))
                self.save_as_json(noManual_full_Feature,
                                  os.path.join(self.args.dataset,
                                               'feature/' + self.args.model_name + '/noManual_full_Feature.json'))
                self.save_as_json(noManual_sorted_score_list,
                                  os.path.join(self.args.dataset,
                                               'results/' + self.args.model_name + '/noManual_sorted_score_list.json'))

        if self.args.model_name == 'WaveMix':
            return noManual_results_list, [], noManual_sorted_score_list, [], dfaulo_time
        name2isfault = self.load_json(os.path.join(self.args.dataset, 'train/name2isfault.json'))
        if os.path.exists(
                os.path.join(self.args.dataset, 'results/' + self.args.model_name + '/Manual_results_list.json')) and \
                os.path.exists(os.path.join(self.args.dataset,
                                            'results/' + self.args.model_name + '/Manual_sorted_score_list.json')) and self.args.ablation == 'None':
            print('Manual_results_list.json already exist!')
            Manual_results_list = self.load_json(
                os.path.join(self.args.dataset, 'results/' + self.args.model_name + '/Manual_results_list.json'))
            Manual_sorted_score_list = self.load_json(
                os.path.join(self.args.dataset, 'results/' + self.args.model_name + '/Manual_sorted_score_list.json'))
        else:
            print('start manual iteration')
            Manual_results_list, Manual_sorted_score_list, Update_Susp_time = self.Manual_iteration(
                noManual_results_list, noManual_full_Feature, name2isfault, noManual_sorted_score_list)
            print('manual iteration finished')
            dfaulo_time['Update Susp'] = Update_Susp_time
            if self.args.ablation == 'None':
                self.save_as_json(Manual_results_list,
                                  os.path.join(self.args.dataset,
                                               'results/' + self.args.model_name + '/Manual_results_list.json'))
                self.save_as_json(Manual_sorted_score_list,
                                  os.path.join(self.args.dataset,
                                               'results/' + self.args.model_name + '/Manual_sorted_score_list.json'))
                dfaulo_time['all'] = dfaulo_time['Select Subset'] + dfaulo_time['Mutation&Extraction'] + dfaulo_time[
                    'Initialize Susp'] + dfaulo_time['Update Susp']

                if dfaulo_time['Select Subset'] != -1 and dfaulo_time['Mutation&Extraction'] != -1 and dfaulo_time[
                    'Initialize Susp'] != -1 and dfaulo_time['Update Susp'] != -1:
                    self.save_as_json(dfaulo_time,
                                      os.path.join(self.args.dataset,
                                                   'results/' + self.args.model_name + '/dfaulo_time.json'))
        return noManual_results_list, Manual_results_list, noManual_sorted_score_list, Manual_sorted_score_list, dfaulo_time

    def Feature_Summary(self, data_s):
        model = eval(self.args.model_name)()
        model.load_state_dict(torch.load(self.args.model))
        torch.save(model, os.path.join(self.args.dataset, 'mutmodel/' + self.args.model_name + '/model.pth'))

        Select_Subset_start = time.time()

        if os.path.exists(os.path.join(self.args.dataset, 'dellist/' + self.args.model_name + '/vae_del_list.json')) and \
                os.path.exists(
                    os.path.join(self.args.dataset, 'dellist/' + self.args.model_name + '/km_del_list.json')) and \
                os.path.exists(
                    os.path.join(self.args.dataset, 'dellist/' + self.args.model_name + '/loss_del_list.json')):
            vae_del_list = self.load_json(
                os.path.join(self.args.dataset, 'dellist/' + self.args.model_name + '/vae_del_list.json'))
            km_del_list = self.load_json(
                os.path.join(self.args.dataset, 'dellist/' + self.args.model_name + '/km_del_list.json'))
            loss_del_list = self.load_json(
                os.path.join(self.args.dataset, 'dellist/' + self.args.model_name + '/loss_del_list.json'))
        else:
            vae_del_list, km_del_list, loss_del_list = self.OAL(data_s)
            # save as json file
            self.save_as_json(vae_del_list,
                              os.path.join(self.args.dataset, 'dellist/' + self.args.model_name + '/vae_del_list.json'))
            self.save_as_json(km_del_list,
                              os.path.join(self.args.dataset, 'dellist/' + self.args.model_name + '/km_del_list.json'))
            self.save_as_json(loss_del_list,
                              os.path.join(self.args.dataset,
                                           'dellist/' + self.args.model_name + '/loss_del_list.json'))
        Select_Subset_end = time.time()
        Select_Subset_time = Select_Subset_end - Select_Subset_start

        # mutation
        Mutation_Extraction_start_time = time.time()
        if os.path.exists(os.path.join(self.args.dataset, 'mutmodel/' + self.args.model_name + '/model_vae.pth')):
            model_vae = torch.load(
                os.path.join(self.args.dataset, 'mutmodel/' + self.args.model_name + '/model_vae.pth'))
        else:
            print('mutation start on vae_del_list')
            model_vae = self.mutation(vae_del_list, data_s, 'model_vae.pth')
        if os.path.exists(os.path.join(self.args.dataset, 'mutmodel/' + self.args.model_name + '/model_km.pth')):
            model_km = torch.load(os.path.join(self.args.dataset, 'mutmodel/' + self.args.model_name + '/model_km.pth'))
        else:
            print('mutation start on km_del_list')
            model_km = self.mutation(km_del_list, data_s, 'model_km.pth')
        if os.path.exists(os.path.join(self.args.dataset, 'mutmodel/' + self.args.model_name + '/model_loss.pth')):
            model_loss = torch.load(
                os.path.join(self.args.dataset, 'mutmodel/' + self.args.model_name + '/model_loss.pth'))
        else:
            print('mutation start on loss_del_list')
            model_loss = self.mutation(loss_del_list, data_s, 'model_loss.pth')
        print(os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/org_Loss_list.json'))
        # get feature
        if os.path.exists(os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/image_list.json')) and \
                os.path.exists(os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/gt_list.json')) and \
                os.path.exists(
                    os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/org_SFM_list.json')) and \
                os.path.exists(
                    os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/org_Loss_list.json')) and \
                os.path.exists(
                    os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/vae_SFM_list.json')) and \
                os.path.exists(
                    os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/vae_Loss_list.json')) and \
                os.path.exists(
                    os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/km_SFM_list.json')) and \
                os.path.exists(
                    os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/km_Loss_list.json')) and \
                os.path.exists(
                    os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/loss_SFM_list.json')) and \
                os.path.exists(
                    os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/loss_Loss_list.json')):

            image_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/image_list.json'))
            gt_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/gt_list.json'))
            org_SFM_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/org_SFM_list.json'))
            org_Loss_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/org_Loss_list.json'))
            vae_SFM_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/vae_SFM_list.json'))
            vae_Loss_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/vae_Loss_list.json'))
            km_SFM_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/km_SFM_list.json'))
            km_Loss_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/km_Loss_list.json'))
            loss_SFM_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/loss_SFM_list.json'))
            loss_Loss_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/loss_Loss_list.json'))
        else:
            print('\n model feature extraction')
            ORG, VAE, KM, LOSS = self.get_feature(model, model_vae, model_km, model_loss, data_s)
            image_list, gt_list, org_SFM_list, org_Loss_list = zip(*ORG)
            vae_SFM_list, vae_Loss_list = zip(*VAE)
            km_SFM_list, km_Loss_list = zip(*KM)
            loss_SFM_list, loss_Loss_list = zip(*LOSS)
            # save as json file
            self.save_as_json(image_list,
                              os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/image_list.json'))
            self.save_as_json(gt_list,
                              os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/gt_list.json'))
            self.save_as_json(org_SFM_list,
                              os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/org_SFM_list.json'))
            self.save_as_json(org_Loss_list,
                              os.path.join(self.args.dataset,
                                           'feature/' + self.args.model_name + '/org_Loss_list.json'))
            self.save_as_json(vae_SFM_list,
                              os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/vae_SFM_list.json'))
            self.save_as_json(vae_Loss_list,
                              os.path.join(self.args.dataset,
                                           'feature/' + self.args.model_name + '/vae_Loss_list.json'))
            self.save_as_json(km_SFM_list,
                              os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/km_SFM_list.json'))
            self.save_as_json(km_Loss_list,
                              os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/km_Loss_list.json'))
            self.save_as_json(loss_SFM_list,
                              os.path.join(self.args.dataset,
                                           'feature/' + self.args.model_name + '/loss_SFM_list.json'))
            self.save_as_json(loss_Loss_list,
                              os.path.join(self.args.dataset,
                                           'feature/' + self.args.model_name + '/loss_Loss_list.json'))

        if os.path.exists(os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/random_index.json')):
            random_index = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/random_index.json'))
        else:
            # sample 10 index for each gt(class_num gts)

            random_index = []
            for i in range(self.class_num):
                if self.args.model_name == 'TCDCNN':
                    index = [j for j in range(len(gt_list))]
                else:
                    index = np.where(np.array(gt_list) == i)[0]
                if self.args.model_name == 'TCDCNN':
                    random_index.extend(random.sample(index, 100))
                else:
                    random_index.extend(random.sample(index.tolist(), 10))
            # save as json file
            self.save_as_json(random_index,
                              os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/random_index.json'))

        if os.path.exists(os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/gt_one_hot_list.json')):
            gt_one_hot_list = self.load_json(
                os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/gt_one_hot_list.json'))
        else:
            if self.args.model_name == 'TCDCNN':
                gt_one_hot_list = gt_list
            else:
                # transform gt_list to one-hot
                gt_one_hot = np.zeros((len(gt_list), self.class_num))
                for i, gt in enumerate(gt_list):
                    gt_one_hot[i][gt] = 1
                gt_one_hot_list = gt_one_hot.tolist()
            # save as json file
            self.save_as_json(gt_one_hot_list,
                              os.path.join(self.args.dataset,
                                           'feature/' + self.args.model_name + '/gt_one_hot_list.json'))

        print('\nfeature extraction finished, start to SUMMARY...')
        if self.args.ablation == 'None' or self.args.ablation == 'all' or self.args.ablation == '1%fed':
            Feature = [[*org_SFM, *gt, *vae_SFM, *km_SFM, *loss_SFM, (1 if img in vae_del_list else 0),
                        (1 if img in km_del_list else 0), (1 if img in loss_del_list else 0), org_Loss, vae_Loss,
                        km_Loss,
                        loss_Loss]
                       for img, gt, org_SFM, vae_SFM, km_SFM, loss_SFM, org_Loss, vae_Loss, km_Loss, loss_Loss in
                       zip(image_list, gt_one_hot_list, org_SFM_list, vae_SFM_list, km_SFM_list, loss_SFM_list,
                           org_Loss_list,
                           vae_Loss_list, km_Loss_list, loss_Loss_list)]
        elif self.args.ablation == 'input':
            Feature = [[*org_SFM, *gt, *vae_SFM, (1 if img in vae_del_list else 0), org_Loss, vae_Loss]
                       for img, gt, org_SFM, vae_SFM, org_Loss, vae_Loss in
                       zip(image_list, gt_one_hot_list, org_SFM_list, vae_SFM_list,
                           org_Loss_list,
                           vae_Loss_list)]
        elif self.args.ablation == 'hidden':
            Feature = [[*org_SFM, *gt, *km_SFM, (1 if img in km_del_list else 0), org_Loss, km_Loss]
                       for img, gt, org_SFM, km_SFM, org_Loss, km_Loss in
                       zip(image_list, gt_one_hot_list, org_SFM_list, km_SFM_list,
                           org_Loss_list,
                           km_Loss_list)]
        elif self.args.ablation == 'output':
            Feature = [[*org_SFM, *gt, *loss_SFM, (1 if img in loss_del_list else 0), org_Loss, loss_Loss]
                       for img, gt, org_SFM, loss_SFM, org_Loss, loss_Loss in
                       zip(image_list, gt_one_hot_list, org_SFM_list, loss_SFM_list,
                           org_Loss_list,
                           loss_Loss_list)]

        print('SUMMARY finished')
        Mutation_Extraction_end_time = time.time()
        Mutation_Extraction_time = Mutation_Extraction_end_time - Mutation_Extraction_start_time
        return Feature, Select_Subset_time, Mutation_Extraction_time

    def Iteration(self, Feature):
        Initialize_Susp_start_time = time.time()
        Feature = np.array(Feature)
        print('Feature shape: ', Feature.shape)
        random_index = self.load_json(
            os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/random_index.json'))
        sample_feature = Feature[random_index]
        print('sample_feature shape: ', sample_feature.shape)

        # random shuffle Feature and corresponding image_list
        image_list = self.load_json(
            os.path.join(self.args.dataset, 'feature/' + self.args.model_name + '/image_list.json'))
        image_list = np.array(image_list)
        idx_shuffle = [i for i in range(len(Feature))]
        # random.seed(2023)
        random.shuffle(idx_shuffle)
        Feature = Feature[idx_shuffle]
        image_list = image_list[idx_shuffle]
        model = torch.load(os.path.join(self.args.dataset, 'mutmodel/' + self.args.model_name + '/model.pth'))
        model_vae = torch.load(os.path.join(self.args.dataset, 'mutmodel/' + self.args.model_name + '/model_vae.pth'))
        model_km = torch.load(os.path.join(self.args.dataset, 'mutmodel/' + self.args.model_name + '/model_km.pth'))
        model_loss = torch.load(os.path.join(self.args.dataset, 'mutmodel/' + self.args.model_name + '/model_loss.pth'))
        random_Feature = self.getrandomfeature(model, model_vae, model_km, model_loss, self.class_num)
        random_Feature = np.array(random_Feature)

        print('random Feature shape:', random_Feature.shape)

        sample_feature = np.concatenate((sample_feature, random_Feature), axis=0)
        print('sample_feature merged shape: ', Feature.shape)
        if self.args.model_name == 'TCDCNN':
            Y = [0 for i in range(100)]
            Y.extend([1 for i in range(100)])
        else:
            Y = [0 for i in range(10 * self.class_num)]  # ground truth: 10*class_num = 0 , class_num = 1
            Y.extend([1 for i in range(self.class_num)])

        Y = np.array(Y)
        print('Y shape: ', Y.shape)

        lg = LogisticRegression(C=1.0)
        lg.fit(sample_feature, Y)

        LRres = lg.predict_proba(Feature)  ####@@@@
        LRres = LRres[:, 1]

        print('LRres shape: ', LRres.shape)

        # sort image_list and Feature by LRres in descending order
        idx = np.argsort(-LRres)
        LRres = LRres[idx]
        image_list = image_list[idx]
        Feature = Feature[idx]
        if self.args.model_name != 'TCDCNN':
            for i in range(len(image_list)):
                image_list[i] = image_list[i].split('\\')[-1]
        sorted_score_list = LRres.tolist()

        Initialize_Susp_end_time = time.time()
        Initialize_Susp_time = Initialize_Susp_end_time - Initialize_Susp_start_time

        return image_list.tolist(), Feature.tolist(), sorted_score_list, Initialize_Susp_time

    def Manual_iteration(self, image_list, Feature, name2isfault, noManual_sorted_score_list):
        Update_Susp_start_time = time.time()
        warnings.filterwarnings("ignore")
        if self.args.ablation == '1%fed':
            check_ratio = 0.01
        else:
            check_ratio = 0.20
        if self.args.ablation == '1%fed' and self.args.model_name=='TCDCNN':
            per_check = 10
        else:
            per_check = 200
        epoches = int(len(image_list) * check_ratio / per_check)
        Feature_left = np.array(Feature).astype('float32')
        print('feature shape: ', Feature_left.shape)
        image_list_left = np.array(image_list)
        noManual_sorted_score_list_left = np.array(noManual_sorted_score_list)

        ground_truth_left = []
        for img in image_list:
            ground_truth_left.append(1 if name2isfault[img] else 0)

        ground_truth_left = np.array(ground_truth_left).astype('int')

        Feature_accumulation = None
        image_list_accumulation = None
        ground_truth_accumulation = None
        sorted_score_accumulation = None

        for epoch in range(epoches):

            Feature_now = Feature_left[:per_check]
            image_list_now = image_list_left[:per_check]
            ground_truth_now = ground_truth_left[:per_check]
            sorted_score_now = noManual_sorted_score_list_left[:per_check]

            Feature_left = Feature_left[per_check:]
            image_list_left = image_list_left[per_check:]
            ground_truth_left = ground_truth_left[per_check:]
            noManual_sorted_score_list_left = noManual_sorted_score_list_left[per_check:]

            IS_LACK = False
            LACK_Feature_accumulation = None
            LACK_ground_truth_accumulation = None
            if Feature_accumulation is None:
                Feature_accumulation = Feature_now
                image_list_accumulation = image_list_now
                ground_truth_accumulation = ground_truth_now
                sorted_score_accumulation = sorted_score_now


            else:
                Feature_accumulation = np.vstack((Feature_accumulation, Feature_now))
                image_list_accumulation = np.hstack((image_list_accumulation, image_list_now))
                ground_truth_accumulation = np.hstack((ground_truth_accumulation, ground_truth_now))
                sorted_score_accumulation = np.hstack((sorted_score_accumulation, sorted_score_now))

            # ensure both have label 0 and 1
            if 0 not in ground_truth_accumulation:
                IS_LACK = True
                for _img_ind, _img_name in enumerate(image_list_left):
                    if not name2isfault[_img_name]:
                        LACK_Feature_accumulation = np.vstack((Feature_accumulation, Feature_left[_img_ind]))
                        LACK_ground_truth_accumulation = np.hstack(
                            (ground_truth_accumulation, ground_truth_left[_img_ind]))
                        break
            elif 1 not in ground_truth_accumulation:
                IS_LACK = True
                for _img_ind, _img_name in enumerate(image_list_left):
                    if name2isfault[_img_name]:
                        LACK_Feature_accumulation = np.vstack((Feature_accumulation, Feature_left[_img_ind]))
                        LACK_ground_truth_accumulation = np.hstack(
                            (ground_truth_accumulation, ground_truth_left[_img_ind]))
                        break

            print('\r', 'epoch: ', epoch, '  Feature_accumulation shape: ', Feature_accumulation.shape,
                  '  Feature_left shape: ', Feature_left.shape, end='')
            lg = LogisticRegression(C=1.0)
            if IS_LACK:
                lg.fit(LACK_Feature_accumulation, LACK_ground_truth_accumulation)
            else:
                lg.fit(Feature_accumulation, ground_truth_accumulation)

            LRres = lg.predict_proba(Feature_left)
            LRres = LRres[:, 1]
            idx = np.argsort(-LRres)

            Feature_left = Feature_left[idx]
            image_list_left = image_list_left[idx]
            ground_truth_left = ground_truth_left[idx]
            noManual_sorted_score_list_left = LRres[idx]

            if epoch == epoches - 1:
                Feature_accumulation = np.vstack((Feature_accumulation, Feature_left))
                image_list_accumulation = np.hstack((image_list_accumulation, image_list_left))
                ground_truth_accumulation = np.hstack((ground_truth_accumulation, ground_truth_left))
                sorted_score_accumulation = np.hstack((sorted_score_accumulation, noManual_sorted_score_list_left))
        Update_Susp_end_time = time.time()
        Update_Susp_time = Update_Susp_end_time - Update_Susp_start_time
        return image_list_accumulation.tolist(), sorted_score_accumulation.tolist(), Update_Susp_time

    def OAL(self, data_s):
        args = self.args
        model = eval(self.args.model_name)()
        model.load_state_dict(torch.load(self.args.model))
        modelargs = torch.load(args.model_args)
        loss_fn = modelargs['loss_fn']

        with open(self.args.class_path, 'r') as f:
            classes = json.load(f)
        class_keys = list(classes.keys())
        vae_del_list, km_del_list, loss_del_list = [], [], []

        for specific_label in class_keys:

            dataset_ = dataset(root=args.dataset, classes_path=self.args.class_path, transform=modelargs['transform'],
                               image_size=eval(args.image_size), image_set=args.image_set,
                               specific_label=specific_label, data_s=data_s)
            data_loader = torch.utils.data.DataLoader(dataset_, batch_size=1, shuffle=False, num_workers=0)

            print('\nrunning OAL on label: ', specific_label)

            def get_features_hook(module, input, output):
                global features
                features = output

            moudle = getattr(model, args.hook_layer)
            hook_handle = moudle.register_forward_hook(get_features_hook)
            model.to(device)
            act_features, image_list, Loss_list = [], [], []
            softmax_func = nn.Softmax(dim=1)
            model.eval()
            vae_data = []
            with torch.no_grad():
                for i, data in enumerate(data_loader):
                    images, labels, image_paths = data
                    vae_data.append(images.cpu().numpy().reshape(-1).tolist())
                    if self.args.model_name == 'TCDCNN':
                        out = model(images.float().to(device))
                    else:
                        out = model(images.to(device))
                    labels = labels.to(device)
                    print('\r', 'processing image: ', i, end='')
                    if self.args.model == 'ResNet':
                        act_features.append(F.avg_pool2d(features, 8).cpu().view(-1).numpy().tolist())
                    elif self.args.model == 'TCDCNN':
                        act_features.append(F.hardtanh(features).cpu().view(-1).numpy().tolist())
                    else:
                        act_features.append(features.cpu().view(-1).numpy().tolist())
                    if self.args.model_name == 'TCDCNN':
                        Loss = model.loss([out],
                                          [labels.float()]).cpu().numpy().item()
                    else:
                        Loss = loss_fn(softmax_func(out), labels).cpu().numpy().item()
                    Loss_list.append(Loss)
                    image_list.append(image_paths[0])

            vae_data = np.array(vae_data)
            vae = VAE(epochs=5, verbose=2)
            vae.fit(vae_data)
            vae_decision_scores_ = vae.decision_scores_

            # delete vae_data to save memory
            del vae_data

            clf_kmeans = KMeans(n_clusters=2)
            clf_kmeans.fit(act_features)
            km_label = clf_kmeans.labels_

            # sort image_list according to vae_decision_scores_ in descending order
            zip_list = list(zip(image_list, vae_decision_scores_, km_label, Loss_list))
            zip_list.sort(key=lambda x: x[1], reverse=True)
            image_list, vae_decision_scores_, km_label, Loss_list = zip(*zip_list)

            tmp_vae_del_list = image_list[:int(len(image_list) * float(args.rm_ratio))]
            if sum(km_label) > len(km_label) / 2:
                tmp_km_del_list = [image_list[i] for i in range(len(image_list)) if km_label[i] == 0]
            else:
                tmp_km_del_list = [image_list[i] for i in range(len(image_list)) if km_label[i] == 1]
            zip_list.sort(key=lambda x: x[3], reverse=True)
            image_list, vae_decision_scores_, km_label, Loss_list = zip(*zip_list)
            tmp_loss_del_list = image_list[:int(len(image_list) * float(args.rm_ratio))]
            vae_del_list.extend(tmp_vae_del_list)
            km_del_list.extend(tmp_km_del_list)
            loss_del_list.extend(tmp_loss_del_list)
            print('\n\nvae_del_list: ', len(vae_del_list))
            print('km_del_list: ', len(km_del_list))
            print('loss_del_list: ', len(loss_del_list))

        return vae_del_list, km_del_list, loss_del_list

    def mutation(self, del_list, data_s, save_path):
        args = self.args

        model = eval(self.args.model_name)()
        model.load_state_dict(torch.load(self.args.model))
        modelargs = torch.load(args.model_args)
        loss_fn = nn.CrossEntropyLoss()
        if modelargs['optimizer'] == 'SGD':
            if self.args.model == 'ResNet' or self.args.model == 'VGG':
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            elif self.args.model == 'TCDCNN':
                optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        elif modelargs['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        else:
            raise ValueError('optimizer not supported')
        datasets = dataset(root=args.dataset, classes_path=self.args.class_path, transform=modelargs['transform'],
                           image_size=eval(args.image_size), image_set=args.image_set,
                           ignore_list=del_list, data_s=data_s)
        data_loader = torch.utils.data.DataLoader(datasets, batch_size=args.retrain_bs, shuffle=True, num_workers=0)

        model.to(device)
        model.train()
        for epoch in range(args.retrain_epoch):
            for i, data in enumerate(data_loader):
                images, labels, image_paths = data
                if self.args.model_name == 'TCDCNN':
                    out = model(images.to(device).float())
                else:
                    out = model(images.to(device))
                labels = labels.to(device)
                if self.args.model_name == 'TCDCNN':
                    loss = model.loss([out],
                                      [labels.float()])
                else:
                    loss = loss_fn(out, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('\r', 'epoch: ', epoch, 'processing batch: ', i, end='')
        model.eval()
        dataset_name = args.dataset.split('/')[-1]
        if self.args.model_name != 'TCDCNN':
            test_data_s = data_slice(self.args, './dataset/OriginalTestData/' + dataset_name + '/test')[0]
        else:
            test_data_s = [None]
        test_data = dataset(root='./dataset/OriginalTestData/' + dataset_name,
                            classes_path=self.args.class_path,
                            transform=modelargs['transform'],
                            image_size=eval(args.image_size),
                            image_set='test',
                            specific_label=None,
                            ignore_list=[],
                            data_s=test_data_s)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
        with torch.no_grad():
            correct = 0
            total = 0
            mse_list = []
            for data in test_loader:
                images, labels, _ = data
                if self.args.model_name == 'TCDCNN':
                    outputs = model(images.float().to(device))
                else:
                    outputs = model(images.to(device))
                if self.args.model_name == 'TCDCNN':
                    accuracy = model.accuracy(outputs, labels.float().to(device))
                    mse_list.append(accuracy)
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    labels = labels.to(device)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            if self.args.model_name == 'TCDCNN':
                print('Test Accuracy before mutation on test set: {} %'.format(sum(mse_list) / len(mse_list)))
            else:
                print('Test Accuracy after mutation on test set: {} %'.format(correct / total))
        print('mutation done!')
        # save model
        torch.save(model, os.path.join(args.dataset, 'mutmodel/' + args.model_name + '/' + save_path))
        return model

    def get_feature(self, model, model_vae, model_km, model_loss, data_s):
        args = self.args
        modelargs = torch.load(args.model_args)
        datasets = dataset(root=args.dataset, classes_path=self.args.class_path, transform=modelargs['transform'],
                           image_size=eval(args.image_size), image_set=args.image_set, data_s=data_s)
        data_loader = torch.utils.data.DataLoader(datasets, batch_size=1, shuffle=False, num_workers=0)
        loss_fn = modelargs['loss_fn']
        softmax_func = nn.Softmax(dim=1)
        model.eval()
        model_vae.eval()
        model_km.eval()
        model_loss.eval()

        model.to(device)
        model_vae.to(device)
        model_km.to(device)
        model_loss.to(device)

        org_SFM_list, org_Loss_list, image_list, gt_list = [], [], [], []
        vae_SFM_list, vae_Loss_list = [], []
        km_SFM_list, km_Loss_list = [], []
        loss_SFM_list, loss_Loss_list = [], []
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                images, labels, image_paths = data
                print('\r', 'get feature processing: ', i, end='')
                if self.args.model_name == 'TCDCNN':
                    images = images.float()
                    labels = labels.float()
                org_out = model(images.to(device))
                vae_out = model_vae(images.to(device))
                km_out = model_km(images.to(device))
                loss_out = model_loss(images.to(device))

                # Loss
                labels = labels.to(device)
                if self.args.model_name == 'TCDCNN':
                    org_Loss = model.loss([org_out], [labels]).cpu().numpy()
                    vae_Loss = model.loss([vae_out], [labels]).cpu().numpy()
                    km_Loss = model.loss([km_out], [labels]).cpu().numpy()
                    loss_Loss = model.loss([loss_out], [labels]).cpu().numpy()
                else:
                    org_Loss = loss_fn(softmax_func(org_out), labels).cpu().numpy()
                    vae_Loss = loss_fn(softmax_func(vae_out), labels).cpu().numpy()
                    km_Loss = loss_fn(softmax_func(km_out), labels).cpu().numpy()
                    loss_Loss = loss_fn(softmax_func(loss_out), labels).cpu().numpy()

                # SoftMax
                if self.args.model_name == 'TCDCNN':
                    org_SFM = org_out.cpu().numpy()[0]
                    vae_SFM = vae_out.cpu().numpy()[0]
                    km_SFM = km_out.cpu().numpy()[0]
                    loss_SFM = loss_out.cpu().numpy()[0]
                else:
                    org_SFM = softmax_func(org_out).cpu().numpy()[0]
                    vae_SFM = softmax_func(vae_out).cpu().numpy()[0]
                    km_SFM = softmax_func(km_out).cpu().numpy()[0]
                    loss_SFM = softmax_func(loss_out).cpu().numpy()[0]

                # append Loss
                org_Loss_list.append(org_Loss.item())
                vae_Loss_list.append(vae_Loss.item())
                km_Loss_list.append(km_Loss.item())
                loss_Loss_list.append(loss_Loss.item())

                # append SoftMax
                org_SFM_list.append(org_SFM.tolist())
                vae_SFM_list.append(vae_SFM.tolist())
                km_SFM_list.append(km_SFM.tolist())
                loss_SFM_list.append(loss_SFM.tolist())
                if self.args.model_name == 'TCDCNN':
                    gt_list.append(labels.cpu().numpy()[0].tolist())
                else:
                    gt_list.append(int(labels.cpu().numpy()[0]))
                image_list.append(image_paths[0])

        return zip(image_list, gt_list, org_SFM_list, org_Loss_list), zip(vae_SFM_list, vae_Loss_list), zip(km_SFM_list,
                                                                                                            km_Loss_list), zip(
            loss_SFM_list, loss_Loss_list)

    def getrandomfeature(self, model_org, model_vae, model_km, model_loss, class_num):
        modelargs = torch.load(self.args.model_args)

        def model_out(model, X, Y):
            model.to(device)
            model.eval()
            loss_fn = modelargs['loss_fn']
            softmax_func = nn.Softmax(dim=1)

            with torch.no_grad():
                X = X.to(device)
                if self.args.model_name == 'TCDCNN':
                    X = X.float()
                    y = Y.float().to(device)
                else:
                    y = torch.from_numpy(np.array([Y])).long().to(device)
                out = model(X)
                if self.args.model_name == 'TCDCNN':
                    soft_output = out
                    loss = model.loss([out], [y]).cpu().numpy().item()
                    sfout = soft_output.cpu().numpy()[0].tolist()
                else:
                    soft_output = softmax_func(out)
                    loss = loss_fn(soft_output, y).cpu().numpy().item()
                    sfout = soft_output.cpu().numpy()[0].tolist()
            return sfout, loss

        image_size = eval(self.args.image_size)

        if eval(self.args.image_size) == None and self.args.model_name != 'TCDCNN':
            X = torch.randint(0, 95805, (1, 100))
        elif self.args.model_name == 'TCDCNN':
            X = torch.rand(1, 1, 40, 40)

        else:
            X = torch.rand(1, image_size[2], image_size[0], image_size[1])

        Feature = []
        if self.args.model_name == 'TCDCNN':
            class_num = 100
        for i in range(class_num):


            if self.args.model_name == 'TCDCNN':
                label = np.zeros((1, 10))
                for i in range(10):
                    label[0, i] = 0 + (40 - 0) * np.random.random()
                label = label.astype('float64')
                label = torch.from_numpy(label)
            else:
                label = i
            sfout_org, loss_org = model_out(model_org, X, label)
            sfout_vae, loss_vae = model_out(model_vae, X, label)
            sfout_km, loss_km = model_out(model_km, X, label)
            sfout_loss, loss_loss = model_out(model_loss, X, label)
            if self.args.model_name == 'TCDCNN':
                gt = label.cpu().numpy()[0].tolist()
            else:
                gt = np.zeros(class_num)
                gt[i] = 1
                gt = gt.tolist()
            if self.args.ablation == 'None' or self.args.ablation == 'all' or self.args.ablation == '1%fed':
                tmp_feature = [*sfout_org, *gt, *sfout_vae, *sfout_km, *sfout_loss, 1, 1, 1, loss_org, loss_vae,
                               loss_km,
                               loss_loss]
            elif self.args.ablation == 'input':
                tmp_feature = [*sfout_org, *gt, *sfout_vae, 1, loss_org, loss_vae]
            elif self.args.ablation == 'hidden':
                tmp_feature = [*sfout_org, *gt, *sfout_km, 1, loss_org, loss_km]
            elif self.args.ablation == 'output':
                tmp_feature = [*sfout_org, *gt, *sfout_loss, 1, loss_org, loss_loss]
            Feature.append(tmp_feature)
        return Feature

    def save_as_json(self, data, save_path):
        data_json = json.dumps(data, indent=4)
        with open(save_path, 'w') as file:
            file.write(data_json)

    def load_json(self, load_path):
        with open(load_path, 'r') as f:
            data = json.load(f)
        return data


def data_slice(args, path_dir, slice_num=1):
    slice_num = slice_num
    random.seed(2023)
    with open(args.class_path, 'r') as f:
        classes = json.load(f)
    class_keys = list(classes.keys())
    result = {i: {} for i in range(slice_num)}
    for name in class_keys:
        img_list_dir = os.listdir(os.path.join(path_dir, name))
        img_list = []
        for img in img_list_dir:
            path = os.path.join(name, img)
            img_list.append(path)
        random.shuffle(img_list)
        slice_len = len(img_list) // slice_num
        for i in range(slice_num):
            result[i][name] = img_list[i * slice_len:(i + 1) * slice_len]
    print('data slice done with slice num: ', slice_num)
    return result
