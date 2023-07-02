import numpy as np
from matplotlib import pyplot as plt
from utils.dataset import *

cm = plt.cm.get_cmap('tab10')

COLOR = {
    "Theory Best": cm(0),
    "DIF": cm(1),
    "Manual": cm(3),
    "Random": cm(7),
    "DeepGini": cm(2),
    "CleanLab": cm(4),
    'Uncertainty': cm(5),
    'DeepState': cm(6),
    'SimiFeat': cm(8),
    'NCNV': cm(9),
}

datasetname = 'MTFL'
ModelTypeList = ['TCDCNN']
NoiseTypeList = ['RandomLabelNoise', 'RandomDataNoise']
MethodTypeList = ['Theory Best','Uncertainty', 'Random','Manual']
for modeltype in ModelTypeList:
    for noisetype in NoiseTypeList:
        plt.figure()
        for method in MethodTypeList:
            if method == 'Theory Best':
                imagelist = load_json(
                    './dataset/' + noisetype + '/' + datasetname + '/results/' + modeltype + '/Manual_results_list.json')
            else:
                imagelist = load_json(
                    './dataset/' + noisetype + '/' + datasetname + '/results/' + modeltype + '/' + method + '_results_list.json')
            name2isfault = load_json('./dataset/' + noisetype + '/' + datasetname + '/train/' + 'name2isfault.json')

            X = []
            Y = []
            Count = 0
            # convert name2isfault values to list

            falutNum = sum(list(name2isfault.values()))

            print('faultNum:', falutNum)
            if method == 'Theory Best':
                the_imagelist = [imagename for imagename in imagelist if name2isfault[imagename]]
                the_imagelist.extend([imagename for imagename in imagelist if not name2isfault[imagename]])
                imagelist = the_imagelist


            for i, imagename in enumerate(imagelist):
                X.append(i/len(imagelist))
                if name2isfault[imagename]:
                    Count += 1
                Y.append(Count/falutNum)
            if method == 'Manual':
                plt.plot(X, Y, color=COLOR[method], label='DfauLo', linewidth=2.0)
            else:
                plt.plot(X, Y, color=COLOR[method], label=method)
        plt.xlabel(r'$Percentage\ of\ test\ case\ executed$')
        plt.ylabel(r'$Percentage\ of\ fault\ detected$')
        plt.legend(loc=4)
        os.makedirs('./dataset/Figure/'+datasetname+'/'+modeltype,exist_ok=True)
        plt.savefig('./dataset/Figure/'+datasetname+'/'+modeltype+'/'+noisetype+'.pdf', bbox_inches='tight')
