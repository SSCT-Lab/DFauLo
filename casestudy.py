# load json
import json
import os

import numpy as np
from PIL import Image

import pandas as pd
from sklearn.linear_model import LogisticRegression


def save_as_json(data, save_path):
    data_json = json.dumps(data, indent=4)
    with open(save_path, 'w') as file:
        file.write(data_json)


def load_json(load_path):
    with open(load_path, 'r') as f:
        data = json.load(f)
    return data


def Manual_Iteration(Round=1):
    # create dir
    RoundSaveRoot = './dataset/CaseStudyData/EMNIST/ManualResults/R' + str(Round)
    os.makedirs(RoundSaveRoot)

    # read excel
    if Round == 6:
        rootpath = './dataset/CaseStudyData/EMNIST/ManualResults/EMNIST_MR6.xlsx'
        # FeaturePath = './dataset/CaseStudyData/EMNIST/feature/WaveMix/noManual_full_Feature.json' # Round1
        FeaturePath = './dataset/CaseStudyData/EMNIST/ManualResults/R5/Feature_left.json' # Round6
        # image_list_path = './dataset/CaseStudyData/EMNIST/results/WaveMix/noManual_results_list.json' # Round1
        image_list_path = './dataset/CaseStudyData/EMNIST/ManualResults/R5/image_list_left.json' # Round6

        # noManual_sorted_score_list_path = './dataset/CaseStudyData/EMNIST/results/WaveMix/noManual_sorted_score_list.json' # Round1
        noManual_sorted_score_list_path = './dataset/CaseStudyData/EMNIST/ManualResults/R5/noManual_sorted_score_list_left.json' # Round6
        df = pd.read_excel(rootpath, sheet_name='dfaulo')
        image_name = df['image_name'].values.tolist()[:200]
        score_num = df['score_num'].values.tolist()[:200]
        print(image_name)
        image2gt = {name.split('_')[-1]: True if num >= 3 else False for name, num in zip(image_name, score_num)}
        print(image2gt)
        END_MANUAL = False
        # Feature_accumulation = None  # if Round == 1 else Feature_accumulation need to be loaded # Round1
        Feature_accumulation = load_json('./dataset/CaseStudyData/EMNIST/ManualResults/R5/Feature_accumulation.json') # Round6
        # image_list_accumulation = None # Round1
        image_list_accumulation = load_json('./dataset/CaseStudyData/EMNIST/ManualResults/R5/image_list_accumulation.json') # Round6
        # ground_truth_accumulation = None # Round1
        ground_truth_accumulation = load_json('./dataset/CaseStudyData/EMNIST/ManualResults/R5/ground_truth_accumulation.json') # Round6
        # sorted_score_accumulation = None # Round1
        sorted_score_accumulation = load_json('./dataset/CaseStudyData/EMNIST/ManualResults/R5/sorted_score_accumulation.json') # Round6

        Feature = load_json(FeaturePath)
        image_list = load_json(image_list_path)
        noManual_sorted_score_list = load_json(noManual_sorted_score_list_path)

        Feature_left = np.array(Feature).astype('float32')  # if Round == 1 else Feature_left need to be loaded
        print('feature shape: ', Feature_left.shape)
        image_list_left = np.array(image_list)  # if Round == 1 else image_list_left need to be loaded
        noManual_sorted_score_list_left = np.array(noManual_sorted_score_list) # if Round == 1 else noManual_sorted_score_list_left need to be loaded

        ground_truth_left = []  # it is new for each round
        for img in image_list[:200]:
            ground_truth_left.append(1 if image2gt[img] else 0)

    ground_truth_left = np.array(ground_truth_left).astype('int')
    per_check = 200

    Feature_now = Feature_left[:per_check]
    image_list_now = image_list_left[:per_check]
    ground_truth_now = ground_truth_left[:per_check]
    sorted_score_now = noManual_sorted_score_list_left[:per_check]

    Feature_left = Feature_left[per_check:]
    image_list_left = image_list_left[per_check:]
    # ground_truth_left = ground_truth_left[per_check:] # no need because it is new for each round
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
    print('accumulation shape: ', Feature_accumulation.shape)
    print(ground_truth_accumulation.shape)
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
    # ground_truth_left = ground_truth_left[idx] # no need because it is new for each round
    noManual_sorted_score_list_left = LRres[idx]

    if END_MANUAL:
        Feature_accumulation = np.vstack((Feature_accumulation, Feature_left))
        image_list_accumulation = np.hstack((image_list_accumulation, image_list_left))
        # ground_truth_accumulation = np.hstack((ground_truth_accumulation, ground_truth_left)) # no need because it is new for each round
        sorted_score_accumulation = np.hstack((sorted_score_accumulation, noManual_sorted_score_list_left))

    save_as_json(Feature_left.tolist(), RoundSaveRoot + '/Feature_left.json')
    save_as_json(image_list_left.tolist(), RoundSaveRoot + '/image_list_left.json')
    # save_as_json(ground_truth_left, RoundSaveRoot + '/ground_truth_left.json') # no need because it is new for each round
    save_as_json(noManual_sorted_score_list_left.tolist(), RoundSaveRoot + '/noManual_sorted_score_list_left.json')

    save_as_json(Feature_accumulation.tolist(), RoundSaveRoot + '/Feature_accumulation.json')
    save_as_json(image_list_accumulation.tolist(), RoundSaveRoot + '/image_list_accumulation.json')
    save_as_json(ground_truth_accumulation.tolist(), RoundSaveRoot + '/ground_truth_accumulation.json')
    save_as_json(sorted_score_accumulation.tolist(), RoundSaveRoot + '/sorted_score_accumulation.json')

def Manual_write2excel(Round = 1):
    RoundSaveRoot = './dataset/CaseStudyData/EMNIST/ManualResults/R' + str(Round)
    results_list = load_json(RoundSaveRoot + '/image_list_left.json')
    sorted_score_list = load_json(RoundSaveRoot + '/noManual_sorted_score_list_left.json')
    # list dir
    rootpath = './dataset/CaseStudyData/EMNIST/'
    labels = os.listdir(rootpath + 'train')
    print(labels)
    name2label = {}
    for label in labels:
        image_names = os.listdir(rootpath + 'train/' + label)
        for image_name in image_names:
            name2label[image_name] = label

    # create a xls file
    import xlwt

    workbook = xlwt.Workbook(encoding='utf-8')


    SAVE_NAME = 'DfauLo'

    worksheet = workbook.add_sheet(SAVE_NAME)

    worksheet.write(0, 0, 'image_name')
    worksheet.write(0, 1, 'label')
    worksheet.write(0, 2, 'prob')
    for i in range(200):
        new_name = str(i) + '_' + 'label_' + name2label[results_list[i]] + '_index_' + results_list[i]

        # open image
        image = Image.open(rootpath + 'train/' + name2label[results_list[i]] + '/' + results_list[i])

        # write new_name and label to xls
        worksheet.write(i + 1, 0, new_name)
        worksheet.write(i + 1, 1, name2label[results_list[i]])
        # save score as .4f
        worksheet.write(i + 1, 2, '%.4f' % sorted_score_list[i])

        # save image as new_name
        if os.path.exists(RoundSaveRoot + '/NextRoundData') is False:
            os.makedirs(RoundSaveRoot + '/NextRoundData')
        image.save(RoundSaveRoot + '/NextRoundData' + '/' + new_name + '.png')

    workbook.save(RoundSaveRoot + '/NextRoundData' + '.xls')


Manual_Iteration(Round = 6)
Manual_write2excel(Round = 6)



def write2excel():
    rootpath = './dataset/CaseStudyData/EMNIST/'

    METHOD = 'Random'

    with open(rootpath + 'results/WaveMix/' + METHOD + '_results_list.json', 'r') as f:
        results_list = json.load(f)
    with open(rootpath + 'results/WaveMix/' + METHOD + '_sorted_score_list.json', 'r') as f:
        sorted_score_list = json.load(f)

    # list dir
    labels = os.listdir(rootpath + 'train')
    print(labels)
    name2label = {}
    for label in labels:
        image_names = os.listdir(rootpath + 'train/' + label)
        for image_name in image_names:
            name2label[image_name] = label

    # create a xls file
    import xlwt

    workbook = xlwt.Workbook(encoding='utf-8')

    SAVE_NAME = METHOD
    if METHOD == 'noManual':
        SAVE_NAME = 'DfauLo'

    worksheet = workbook.add_sheet(SAVE_NAME)

    worksheet.write(0, 0, 'image_name')
    worksheet.write(0, 1, 'label')
    worksheet.write(0, 2, 'prob')
    for i in range(int(len(results_list) * 0.01)):
        new_name = str(i) + '_' + 'label_' + name2label[results_list[i]] + '_index_' + results_list[i]

        # open image
        image = Image.open(rootpath + 'train/' + name2label[results_list[i]] + '/' + results_list[i])

        # write new_name and label to xls
        worksheet.write(i + 1, 0, new_name)
        worksheet.write(i + 1, 1, name2label[results_list[i]])
        # save score as .4f
        worksheet.write(i + 1, 2, '%.4f' % sorted_score_list[i])

        # save image as new_name
        if os.path.exists(rootpath + 'Manual/' + SAVE_NAME) is False:
            os.makedirs(rootpath + 'Manual/' + SAVE_NAME)
        image.save(rootpath + 'Manual/' + SAVE_NAME + '/' + new_name + '.png')

    workbook.save(rootpath + 'Manual/' + SAVE_NAME + '.xls')
