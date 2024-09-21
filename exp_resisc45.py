import argparse
import random

import numpy as np

from dfaulo import *
from utils.evaluation import *

NoiseTypeList = ['RandomLabelNoise']

for NoiseType in NoiseTypeList:
    DataSet = 'RESISC45'
    # NoiseType = 'SpecificLabelNoise'
    Model = 'WaveMix'
    hook_layer = 'pool' #conv
    image_size = '(256,256,3)'
    random.seed(2023)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='./dataset/' + NoiseType + '/' + DataSet, help='input dataset')
    parser.add_argument('--model', default='./dataset/' + NoiseType + '/' + DataSet + '/' + Model + '.pth',
                        help='input model path')
    parser.add_argument('--model_name', default=Model, help='input model path')
    parser.add_argument('--class_path', default='./dataset/' + DataSet.lower() + '_classes.json',
                        help='input model path')
    parser.add_argument('--image_size', default=image_size, help='input image size')
    parser.add_argument('--model_args', default='./dataset/' + DataSet.lower() + '_model_args.pth',
                        help='input model args path')
    parser.add_argument('--image_set', default='train', help='input image set')
    parser.add_argument('--hook_layer', default=hook_layer, help='input hook layer')
    parser.add_argument('--rm_ratio', default=0.05, help='input ratio')
    parser.add_argument('--retrain_epoch', default=10, help='input retrain epoch')
    parser.add_argument('--retrain_bs', default=64, help='input retrain batch size')
    parser.add_argument('--slice_num', default=1, help='input slice num')
    parser.add_argument('--ablation', default='None', help='input slice num')

    args = parser.parse_args()
    args.slice_num = int(args.slice_num)
    args.rm_ratio = float(args.rm_ratio)
    args.retrain_epoch = int(args.retrain_epoch)
    args.retrain_bs = int(args.retrain_bs)
    if DataSet!='MTFL':
        data_s = data_slice(args, args.dataset + '/' + args.image_set, args.slice_num)
    else:
        data_s = [None]

    results = []
    df = DfauLo(args)

    noManual_results_list, Manual_results_list, noManual_sorted_score_list, Manual_sorted_score_list, dfaulo_time = df.run(
        data_s[0])
    print('dfaulo time: ', dfaulo_time)

    if NoiseType == 'CaseStudyData':
        exit(0)

    with open(os.path.join(args.dataset, 'train/name2isfault.json'), 'r') as f:
        name2isfault = json.load(f)
    ### POBL10
    noManual_pobl10 = POBL(noManual_results_list, name2isfault, 0.1)
    print('\nnoManual POBL10: ', noManual_pobl10)
    Manual_pobl10 = POBL(Manual_results_list, name2isfault, 0.1)
    print('Manual POBL10: ', Manual_pobl10)


    ### APFD
    noManual_apfd = APFD(noManual_results_list, name2isfault)
    print('\nnoManual APFD: ', noManual_apfd)
    Manual_apfd = APFD(Manual_results_list, name2isfault)
    print('Manual APFD: ', Manual_apfd)

    ### RAUC
    noManual_rauc = RAUC(noManual_results_list, name2isfault)
    print('\nnoManual RAUC: ', noManual_rauc)
    Manual_rauc = RAUC(Manual_results_list, name2isfault)
    print('Manual RAUC: ', Manual_rauc)

    ### ROC_AUC
    noManual_roc_auc = ROC_AUC(noManual_results_list, name2isfault, noManual_sorted_score_list)
    print('\nnoManual ROC_AUC: ', noManual_roc_auc)
    Manual_roc_auc = ROC_AUC(Manual_results_list, name2isfault, Manual_sorted_score_list)
    print('Manual ROC_AUC: ', Manual_roc_auc)
