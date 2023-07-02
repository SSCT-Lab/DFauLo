import argparse
import random

import numpy as np

from dfaulo import *
from baselines import *
from utils.evaluation import *
from scipy.stats import ranksums
from cliffs_delta import cliffs_delta


def check(a, b):
    ans = ''
    s, p = ranksums(a, b)
    d, res = cliffs_delta(a, b)
    if p >= 0.05:
        ans = 'T'
    elif p < 0.05:
        if d >= 0.147:
            ans = 'W'
        elif d <= -0.147:
            ans = 'L'
        else:
            ans = 'T'
    return ans





#///////////////////////////////

NoiseTypeList = ['RandomLabelNoise', 'RandomDataNoise']
AblationList = ['all', 'input', 'hidden', 'output', '1%fed']
for NoiseType in NoiseTypeList:
    Ablation_Result = {}
    for AblationType in AblationList:
        DataSet = 'MTFL'
        # NoiseType = 'SpecificLabelNoise'
        Model = 'TCDCNN'
        hook_layer = 'conv4'
        image_size = 'None'

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
        parser.add_argument('--ablation', default=AblationType, help='input slice num')

        args = parser.parse_args()
        args.slice_num = int(args.slice_num)
        args.rm_ratio = float(args.rm_ratio)
        args.retrain_epoch = int(args.retrain_epoch)
        args.retrain_bs = int(args.retrain_bs)

        if args.model_name == 'TCDCNN':
            data_s=[None]
        else:
            data_s = data_slice(args, args.dataset + '/' + args.image_set, args.slice_num)

        results = []
        df = DfauLo(args)
        Ablation_Result[AblationType] = {}
        Ablation_Result[AblationType]['noManual_results_list'] = []
        Ablation_Result[AblationType]['Manual_results_list'] = []
        Ablation_Result[AblationType]['noManual_rauc'] = []
        Ablation_Result[AblationType]['Manual_rauc'] = []
        Ablation_Result[AblationType]['noManual_sorted_score_list'] = []
        Ablation_Result[AblationType]['Manual_sorted_score_list'] = []

        with open(os.path.join(args.dataset, 'train/name2isfault.json'), 'r') as f:
            name2isfault = json.load(f)
        for _ in range(30):
            noManual_results_list, Manual_results_list, noManual_sorted_score_list, Manual_sorted_score_list, dfaulo_time = df.run(
                data_s[0])
            noManual_rauc = RAUC(noManual_results_list, name2isfault)
            Manual_rauc = RAUC(Manual_results_list, name2isfault)
            Ablation_Result[AblationType]['noManual_results_list'].append(noManual_results_list)
            Ablation_Result[AblationType]['Manual_results_list'].append(Manual_results_list)
            Ablation_Result[AblationType]['noManual_rauc'].append(noManual_rauc)
            Ablation_Result[AblationType]['Manual_rauc'].append(Manual_rauc)
            Ablation_Result[AblationType]['noManual_sorted_score_list'].append(noManual_sorted_score_list)
            Ablation_Result[AblationType]['Manual_sorted_score_list'].append(Manual_sorted_score_list)

    for AblationType in AblationList:
        noManualcmp = check(Ablation_Result['all']['Manual_rauc'], Ablation_Result[AblationType]['noManual_rauc'])
        Manualcmp = check(Ablation_Result['all']['Manual_rauc'], Ablation_Result[AblationType]['Manual_rauc'])

        Ablation_Result[AblationType]['noManualcmp'] = noManualcmp
        Ablation_Result[AblationType]['Manualcmp'] = Manualcmp

    data_json = json.dumps(Ablation_Result, indent=4)
    with open(os.path.join(args.dataset, 'results/' + args.model_name + '/Ablation_Result.json'), 'w') as file:
        file.write(data_json)


