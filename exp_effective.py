import argparse
import random

import numpy as np

from dfaulo import *
from baselines import *
from utils.evaluation import *

NoiseTypeList = ['RandomLabelNoise','RandomDataNoise']

for NoiseType in NoiseTypeList:
    DataSet = 'MTFL'
    # NoiseType = 'SpecificLabelNoise'
    Model = 'TCDCNN'
    hook_layer = 'conv4'
    image_size = 'None'
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
    bs = Baselines(args)

    noManual_results_list, Manual_results_list, noManual_sorted_score_list, Manual_sorted_score_list, dfaulo_time = df.run(
        data_s[0])
    print('dfaulo time: ', dfaulo_time)
    if DataSet!='MTFL':
        DIF_results_list, DIF_sorted_score_list, DIF_time = bs.DIF(data_s[0])
        print('DIF time: ', DIF_time)
        SimiFeat_results_list, SimiFeat_sorted_score_list, SimiFeat_time = bs.SimiFeat(data_s[0])
        print('SimiFeat time: ', SimiFeat_time)
        NCNV_results_list, NCNV_sorted_score_list, NCNV_time = bs.NCNV(data_s[0])
        print('NCNV time: ', NCNV_time)
        DeepGini_results_list, DeepGini_sorted_score_list, DeepGini_time = bs.DeepGini(data_s[0])
        print('DeepGini time: ', DeepGini_time)
        CleanLab_results_list, CleanLab_sorted_score_list, CleanLab_time = bs.CleanLab(data_s[0])
        print('CleanLab time: ', CleanLab_time)
    Random_results_list, Random_sorted_score_list, Random_time = bs.Random()
    print('Random time: ', Random_time)
    if DataSet == 'AgNews':
        DeepState_results_list, DeepState_sorted_score_list, DeepState_time = bs.DeepState(data_s[0])
        print('DeepState time: ', DeepState_time)
    if Model == 'VGG' or Model == 'TCDCNN':
        Uncertainty_results_list, Uncertainty_sorted_score_list, Uncertainty_time = bs.Uncertainty(data_s[0])
        print('Uncertainty time: ', Uncertainty_time)

    if NoiseType == 'CaseStudyData':
        exit(0)

    with open(os.path.join(args.dataset, 'train/name2isfault.json'), 'r') as f:
        name2isfault = json.load(f)
    ### POBL10
    noManual_pobl10 = POBL(noManual_results_list, name2isfault, 0.1)
    print('\nnoManual POBL10: ', noManual_pobl10)
    Manual_pobl10 = POBL(Manual_results_list, name2isfault, 0.1)
    print('Manual POBL10: ', Manual_pobl10)

    if DataSet!='MTFL':
        DIF_pobl10 = POBL(DIF_results_list, name2isfault, 0.1)
        print('DIF POBL10: ', DIF_pobl10)
        SimiFeat_pobl10 = POBL(SimiFeat_results_list, name2isfault, 0.1)
        print('SimiFeat POBL10: ', SimiFeat_pobl10)
        NCNV_pobl10 = POBL(NCNV_results_list, name2isfault, 0.1)
        print('NCNV POBL10: ', NCNV_pobl10)
        DeepGini_pobl10 = POBL(DeepGini_results_list, name2isfault, 0.1)
        print('DeepGini POBL10: ', DeepGini_pobl10)
        CleanLab_pobl10 = POBL(CleanLab_results_list, name2isfault, 0.1)
        print('CleanLab POBL10: ', CleanLab_pobl10)
    Random_pobl10 = POBL(Random_results_list, name2isfault, 0.1)
    print('Random POBL10: ', Random_pobl10)
    if DataSet == 'AgNews':
        DeepState_pobl10 = POBL(DeepState_results_list, name2isfault, 0.1)
        print('DeepState POBL10: ', DeepState_pobl10)
    if Model == 'VGG' or Model == 'TCDCNN':
        Uncertainty_pobl10 = POBL(Uncertainty_results_list, name2isfault, 0.1)
        print('Uncertainty POBL10: ', Uncertainty_pobl10)

    ### APFD
    noManual_apfd = APFD(noManual_results_list, name2isfault)
    print('\nnoManual APFD: ', noManual_apfd)
    Manual_apfd = APFD(Manual_results_list, name2isfault)
    print('Manual APFD: ', Manual_apfd)
    if DataSet!='MTFL':
        DIF_apfd = APFD(DIF_results_list, name2isfault)
        print('DIF APFD: ', DIF_apfd)
        SimiFeat_apfd = APFD(SimiFeat_results_list, name2isfault)
        print('SimiFeat APFD: ', SimiFeat_apfd)
        NCNV_apfd = APFD(NCNV_results_list, name2isfault)
        print('NCNV APFD: ', NCNV_apfd)
        DeepGini_apfd = APFD(DeepGini_results_list, name2isfault)
        print('DeepGini APFD: ', DeepGini_apfd)
        CleanLab_apfd = APFD(CleanLab_results_list, name2isfault)
        print('CleanLab APFD: ', CleanLab_apfd)
    Random_apfd = APFD(Random_results_list, name2isfault)
    print('Random APFD: ', Random_apfd)
    if DataSet == 'AgNews':
        DeepState_apfd = APFD(DeepState_results_list, name2isfault)
        print('DeepState APFD: ', DeepState_apfd)
    if Model == 'VGG' or Model == 'TCDCNN':
        Uncertainty_apfd = APFD(Uncertainty_results_list, name2isfault)
        print('Uncertainty APFD: ', Uncertainty_apfd)

    ### RAUC
    noManual_rauc = RAUC(noManual_results_list, name2isfault)
    print('\nnoManual RAUC: ', noManual_rauc)
    Manual_rauc = RAUC(Manual_results_list, name2isfault)
    print('Manual RAUC: ', Manual_rauc)
    if DataSet!='MTFL':
        DIF_rauc = RAUC(DIF_results_list, name2isfault)
        print('DIF RAUC: ', DIF_rauc)
        SimiFeat_rauc = RAUC(SimiFeat_results_list, name2isfault)
        print('SimiFeat RAUC: ', SimiFeat_rauc)
        NCNV_rauc = RAUC(NCNV_results_list, name2isfault)
        print('NCNV RAUC: ', NCNV_rauc)
        DeepGini_rauc = RAUC(DeepGini_results_list, name2isfault)
        print('DeepGini RAUC: ', DeepGini_rauc)
        CleanLab_rauc = RAUC(CleanLab_results_list, name2isfault)
        print('CleanLab RAUC: ', CleanLab_rauc)
    Random_rauc = RAUC(Random_results_list, name2isfault)
    print('Random RAUC: ', Random_rauc)
    if DataSet == 'AgNews':
        DeepState_rauc = RAUC(DeepState_results_list, name2isfault)
        print('DeepState RAUC: ', DeepState_rauc)
    if Model == 'VGG' or Model == 'TCDCNN':
        Uncertainty_rauc = RAUC(Uncertainty_results_list, name2isfault)
        print('Uncertainty RAUC: ', Uncertainty_rauc)

    ### ROC_AUC
    noManual_roc_auc = ROC_AUC(noManual_results_list, name2isfault, noManual_sorted_score_list)
    print('\nnoManual ROC_AUC: ', noManual_roc_auc)
    Manual_roc_auc = ROC_AUC(Manual_results_list, name2isfault, Manual_sorted_score_list)
    print('Manual ROC_AUC: ', Manual_roc_auc)
    if DataSet!='MTFL':
        DIF_roc_auc = ROC_AUC(DIF_results_list, name2isfault, DIF_sorted_score_list)
        print('DIF ROC_AUC: ', DIF_roc_auc)
        SimiFeat_roc_auc = ROC_AUC(SimiFeat_results_list, name2isfault, SimiFeat_sorted_score_list)
        print('SimiFeat ROC_AUC: ', SimiFeat_roc_auc)
        NCNV_roc_auc = ROC_AUC(NCNV_results_list, name2isfault, NCNV_sorted_score_list)
        print('NCNV ROC_AUC: ', NCNV_roc_auc)
        DeepGini_roc_auc = ROC_AUC(DeepGini_results_list, name2isfault, DeepGini_sorted_score_list)
        print('DeepGini ROC_AUC: ', DeepGini_roc_auc)
        CleanLab_roc_auc = ROC_AUC(CleanLab_results_list, name2isfault, (-np.array(CleanLab_sorted_score_list)).tolist())
        print('CleanLab ROC_AUC: ', CleanLab_roc_auc)
    Random_roc_auc = ROC_AUC(Random_results_list, name2isfault, Random_sorted_score_list)
    print('Random ROC_AUC: ', Random_roc_auc)
    if DataSet == 'AgNews':
        DeepState_roc_auc = ROC_AUC(DeepState_results_list, name2isfault, DeepState_sorted_score_list)
        print('DeepState ROC_AUC: ', DeepState_roc_auc)
    if Model == 'VGG' or Model == 'TCDCNN':
        Uncertainty_roc_auc = ROC_AUC(Uncertainty_results_list, name2isfault, Uncertainty_sorted_score_list)
        print('Uncertainty ROC_AUC: ', Uncertainty_roc_auc)

    print('dfaulo time: ', dfaulo_time)
    if DataSet!='MTFL':
        print('DIF time: ', DIF_time)
        print('SimiFeat time: ', SimiFeat_time)
        print('NCNV time: ', NCNV_time)
        print('DeepGini time: ', DeepGini_time)
        print('CleanLab time: ', CleanLab_time)
    print('Random time: ', Random_time)
    if DataSet == 'AgNews':
        print('DeepState time: ', DeepState_time)
    if DataSet == 'AgNews':
        RESULTSDIR = {
            'noManual': {'pobl10': noManual_pobl10, 'apfd': noManual_apfd, 'rauc': noManual_rauc,
                         'roc_auc': noManual_roc_auc},
            'Manual': {'pobl10': Manual_pobl10, 'apfd': Manual_apfd, 'rauc': Manual_rauc, 'roc_auc': Manual_roc_auc},
            'DIF': {'pobl10': DIF_pobl10, 'apfd': DIF_apfd, 'rauc': DIF_rauc, 'roc_auc': DIF_roc_auc},
            'SimiFeat': {'pobl10': SimiFeat_pobl10, 'apfd': SimiFeat_apfd, 'rauc': SimiFeat_rauc,
                         'roc_auc': SimiFeat_roc_auc},
            'NCNV': {'pobl10': NCNV_pobl10, 'apfd': NCNV_apfd, 'rauc': NCNV_rauc, 'roc_auc': NCNV_roc_auc},
            'DeepGini': {'pobl10': DeepGini_pobl10, 'apfd': DeepGini_apfd, 'rauc': DeepGini_rauc,
                         'roc_auc': DeepGini_roc_auc},
            'CleanLab': {'pobl10': CleanLab_pobl10, 'apfd': CleanLab_apfd, 'rauc': CleanLab_rauc,
                         'roc_auc': CleanLab_roc_auc},
            'Random': {'pobl10': Random_pobl10, 'apfd': Random_apfd, 'rauc': Random_rauc, 'roc_auc': Random_roc_auc},

            'DeepState': {'pobl10': DeepState_pobl10, 'apfd': DeepState_apfd, 'rauc': DeepState_rauc,
                            'roc_auc': DeepState_roc_auc}
        }
    elif Model == 'VGG':
        RESULTSDIR = {
            'noManual': {'pobl10': noManual_pobl10, 'apfd': noManual_apfd, 'rauc': noManual_rauc,
                         'roc_auc': noManual_roc_auc},
            'Manual': {'pobl10': Manual_pobl10, 'apfd': Manual_apfd, 'rauc': Manual_rauc, 'roc_auc': Manual_roc_auc},
            'DIF': {'pobl10': DIF_pobl10, 'apfd': DIF_apfd, 'rauc': DIF_rauc, 'roc_auc': DIF_roc_auc},
            'SimiFeat': {'pobl10': SimiFeat_pobl10, 'apfd': SimiFeat_apfd, 'rauc': SimiFeat_rauc,
                         'roc_auc': SimiFeat_roc_auc},
            'NCNV': {'pobl10': NCNV_pobl10, 'apfd': NCNV_apfd, 'rauc': NCNV_rauc, 'roc_auc': NCNV_roc_auc},
            'DeepGini': {'pobl10': DeepGini_pobl10, 'apfd': DeepGini_apfd, 'rauc': DeepGini_rauc,
                         'roc_auc': DeepGini_roc_auc},
            'CleanLab': {'pobl10': CleanLab_pobl10, 'apfd': CleanLab_apfd, 'rauc': CleanLab_rauc,
                         'roc_auc': CleanLab_roc_auc},
            'Random': {'pobl10': Random_pobl10, 'apfd': Random_apfd, 'rauc': Random_rauc, 'roc_auc': Random_roc_auc},

            'Uncertainty': {'pobl10': Uncertainty_pobl10, 'apfd': Uncertainty_apfd, 'rauc': Uncertainty_rauc,
                            'roc_auc': Uncertainty_roc_auc}
        }
    elif Model == 'TCDCNN':
        RESULTSDIR = {
            'noManual': {'pobl10': noManual_pobl10, 'apfd': noManual_apfd, 'rauc': noManual_rauc,
                         'roc_auc': noManual_roc_auc},
            'Manual': {'pobl10': Manual_pobl10, 'apfd': Manual_apfd, 'rauc': Manual_rauc, 'roc_auc': Manual_roc_auc},
            'Random': {'pobl10': Random_pobl10, 'apfd': Random_apfd, 'rauc': Random_rauc, 'roc_auc': Random_roc_auc},

            'Uncertainty': {'pobl10': Uncertainty_pobl10, 'apfd': Uncertainty_apfd, 'rauc': Uncertainty_rauc,
                            'roc_auc': Uncertainty_roc_auc}
        }
    else:
        RESULTSDIR = {
            'noManual': {'pobl10': noManual_pobl10, 'apfd': noManual_apfd, 'rauc': noManual_rauc,
                         'roc_auc': noManual_roc_auc},
            'Manual': {'pobl10': Manual_pobl10, 'apfd': Manual_apfd, 'rauc': Manual_rauc, 'roc_auc': Manual_roc_auc},
            'DIF': {'pobl10': DIF_pobl10, 'apfd': DIF_apfd, 'rauc': DIF_rauc, 'roc_auc': DIF_roc_auc},
            'SimiFeat': {'pobl10': SimiFeat_pobl10, 'apfd': SimiFeat_apfd, 'rauc': SimiFeat_rauc,
                         'roc_auc': SimiFeat_roc_auc},
            'NCNV': {'pobl10': NCNV_pobl10, 'apfd': NCNV_apfd, 'rauc': NCNV_rauc, 'roc_auc': NCNV_roc_auc},
            'DeepGini': {'pobl10': DeepGini_pobl10, 'apfd': DeepGini_apfd, 'rauc': DeepGini_rauc,
                         'roc_auc': DeepGini_roc_auc},
            'CleanLab': {'pobl10': CleanLab_pobl10, 'apfd': CleanLab_apfd, 'rauc': CleanLab_rauc,
                         'roc_auc': CleanLab_roc_auc},
            'Random': {'pobl10': Random_pobl10, 'apfd': Random_apfd, 'rauc': Random_rauc, 'roc_auc': Random_roc_auc},

        }

    data_json = json.dumps(RESULTSDIR, indent=4)
    with open(os.path.join(args.dataset, 'results/' + args.model_name + '/RESULTSDIR.json'), 'w') as file:
        file.write(data_json)
