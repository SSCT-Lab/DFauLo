# Author: Gentry Atkinson
# Organization: Texas State University
# Data: 08 Sep, 2022
#
# KNN-based noisy label detection based on:
#  https://arxiv.org/abs/2110.06283

# Some of this work has been adapted from: https://github.com/UCSC-REAL/SimiFeat


import numpy as np
import torch
import utils.augmentation
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine
from utils.hoc import get_T_global_min_new, get_score, count_knn_distribution
from torch.utils.data import DataLoader
from scipy import stats

NUM_CLEAN_EPOCHS = 10
NUM_WORKERS = 0
BATCH_SIZE = 32


class CONFIG():
    def __init__(self) -> None:
        self.method = 'rank1'
        self.k = 5
        self.max_iter = 150
        self.G = 10
        self.seed = 1899
        self.num_classes = 0
        self.cnt = 0
        self.min_similarity = 0.0
        self.Tii_offset = 1.0


config = CONFIG()

device = "cuda" if torch.cuda.is_available() else "cpu"

global_dic = {}


################### PyTorch Functions ######################

# (x - y)^2 = x^2 - 2*x*y + y^2
def similarity_matrix(mat: torch.Tensor):
    # get the product x * y
    # here, y = x.t()
    r = torch.mm(mat, mat.t())
    # get the diagonal elements
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    # compute the distance matrix
    D = diag + diag.t() - 2 * r
    return D.sqrt()


def compute_apparent_clusterability_torch(
        fet: torch.Tensor,
        y: torch.Tensor
) -> float:
    """
    Compute that percentage of instances in the feature space that
    share an assigned label with their 2 nearest neighbors
    """
    mat = similarity_matrix(fet)
    # kth value counts from 1
    _, idx_1 = torch.kthvalue(mat, 2, dim=1)
    _, idx_2 = torch.kthvalue(mat, 3, dim=1)
    clusterable_count = 0
    for i in range(idx_1.shape[0]):
        if y[i] == y[idx_1[i]] == y[idx_2[i]]:
            clusterable_count += 1
    return clusterable_count / fet.shape[0]


def setup_dataloader(X: np.ndarray, y: np.ndarray, shuffle=False):
    torch_X = torch.Tensor(X)
    torch_y = torch.Tensor(y)
    torch_index = torch.arange(torch_y.shape[0])

    dataset = torch.utils.data.TensorDataset(torch_X, torch_y, torch_index)
    dataloader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
        drop_last=False, num_workers=NUM_WORKERS
    )
    return dataloader


################### Numpy Functions ######################

def compute_apparent_clusterability(
        fet: np.ndarray,
        y: np.ndarray,
) -> float:
    """
    Compute that percentage of instances in the feature space that
    share an assigned label with their 2 nearest neighbors
    """
    neigh = NearestNeighbors(n_neighbors=3, radius=1.0, metric='minkowski', n_jobs=8)
    neigh.fit(fet)
    clusterable_count = 0
    n = neigh.kneighbors(fet, 3, return_distance=False)
    print('.', end='\n')
    n = np.delete(n, 0, axis=1)
    for i in range(len(fet)):
        if y[i] == y[n[i][0]] == y[n[i][1]]:
            clusterable_count += 1
    print('.', end='\n')

    return clusterable_count / fet.shape[0]


def KNNLabel(
        fet: np.ndarray,
        y: np.ndarray,
        num_classes: int,
        k: int
) -> np.ndarray:
    """
    Return an n x num_classes array of "fuzzy labels"
    for every instance in the feature set based on the
    labels of the k nearest neighbors
    """
    neigh = NearestNeighbors(n_neighbors=k + 1, radius=1.0, metric=cosine)
    neigh.fit(fet)
    fuzzy_y = None
    for x0 in X:
        x0 = np.array([x0])
        nearest = neigh.kneighbors(x0, n_neighbors=None, return_distance=False)[1:]
        n_labels = np.array([np.count_nonzero(nearest == i) for i in range(num_classes)])
        n_labels = np.divide(n_labels, num_classes)
        if fuzzy_y is not None:
            fuzzy_y = np.concatenate((fuzzy_y, np.array([n_labels])), axis=0)
        else:
            fuzzy_y = np.array([n_labels])
    return fuzzy_y


def get_knn_acc_all_class(args, data_set, k=10, noise_prior=None, sel_noisy=None, thre_noise_rate=0.5, thre_true=None):
    # Build Feature Clusters --------------------------------------
    KINDS = args.num_classes

    all_point_cnt = data_set['feature'].shape[0]
    # global
    sample = np.random.choice(np.arange(data_set['feature'].shape[0]), all_point_cnt, replace=False)
    # final_feat, noisy_label = get_feat_clusters(data_set, sample)
    final_feat = data_set['feature'][sample]
    noisy_label = data_set['noisy_label'][sample]
    noise_or_not_sample = data_set['noise_or_not'][sample]
    sel_idx = data_set['index'][sample]
    knn_labels_cnt = count_knn_distribution(args, final_feat, noisy_label, all_point_cnt, k=k, norm='l2')

    method = 'ce'
    # time_score = time.time()
    score = get_score(knn_labels_cnt, noisy_label, k=k, method=method, prior=noise_prior)  # method = ['cores', 'peer']
    # print(f'time for get_score is {time.time()-time_score}')
    score_np = score.cpu().numpy()

    if args.method == 'mv':
        # test majority voting
        print(f'Use MV')
        label_pred = np.argmax(knn_labels_cnt, axis=1).reshape(-1)
        sel_noisy += (sel_idx[label_pred != noisy_label]).tolist()
    elif args.method == 'rank1':
        print(f'Use rank1')
        print(f'Tii offset is {args.Tii_offset}')
        # fig=plt.figure(figsize=(15,4))
        for sel_class in range(KINDS):
            thre_noise_rate_per_class = 1 - min(args.Tii_offset * thre_noise_rate[sel_class][sel_class], 1.0)
            if thre_noise_rate_per_class >= 1.0:
                thre_noise_rate_per_class = 0.95
            elif thre_noise_rate_per_class <= 0.0:
                thre_noise_rate_per_class = 0.05
            sel_labels = (noisy_label.cpu().numpy() == sel_class)
            thre = np.percentile(score_np[sel_labels], 100 * (1 - thre_noise_rate_per_class))

            indicator_all_tail = (score_np >= thre) * (sel_labels)
            sel_noisy += sel_idx[indicator_all_tail].tolist()
    else:
        raise NameError('Undefined method')

    return sel_noisy


# From: UCSC-Real
def data_transform(record, noise_or_not, sel_noisy):
    # assert noise_or_not is not None
    total_len = sum([len(a) for a in record])
    origin_trans = torch.zeros(total_len, record[0][0]['feature'].shape[0])
    origin_label = torch.zeros(total_len).long()
    noise_or_not_reorder = np.empty(total_len, dtype=bool)
    index_rec = np.zeros(total_len, dtype=int)
    cnt, lb = 0, 0
    sel_noisy = np.array(sel_noisy)
    noisy_prior = np.zeros(len(record))

    for item in record:
        for i in item:
            # if i['index'] not in sel_noisy:
            origin_trans[cnt] = i['feature']
            origin_label[cnt] = lb
            noise_or_not_reorder[cnt] = noise_or_not[i['index']] if noise_or_not is not None else False
            index_rec[cnt] = i['index']
            cnt += 1 - np.sum(sel_noisy == i['index'])
            # print(cnt)
        noisy_prior[lb] = cnt - np.sum(noisy_prior)
        lb += 1
    data_set = {'feature': origin_trans[:cnt], 'noisy_label': origin_label[:cnt],
                'noise_or_not': noise_or_not_reorder[:cnt], 'index': index_rec[:cnt]}
    return data_set, noisy_prior / cnt


# From: UCSC-Real
def noniterate_detection(config, record, train_dataset, sel_noisy=[]):
    global global_dic

    T_given_noisy_true = None
    T_given_noisy = None

    # non-iterate
    # sel_noisy = []
    data_set, noisy_prior = data_transform(record, None, sel_noisy)
    # print(data_set['noisy_label'])
    if config.method == 'rank1':
        # T_init = global_var.get_value('T_init')
        # p_init = global_var.get_value('p_init')
        if 'T_init' in global_dic.keys():
            T_init = global_dic['T_init']
        else:
            T_init = None

        if 'p_init' in global_dic.keys():
            p_init = global_dic['p_init']
        else:
            p_init = None

        # print(f'T_init is {T_init}')
        T, p, global_dic = get_T_global_min_new(config, data_set=data_set,
                                                max_step=config.max_iter if T_init is None else 20,
                                                lr=0.1 if T_init is None else 0.01, NumTest=config.G, T0=T_init,
                                                p0=p_init, global_dic=global_dic)

        T_given_noisy = T * p / noisy_prior
        # add randomness
        for i in range(T.shape[0]):
            T_given_noisy[i][i] += np.random.uniform(low=-0.05, high=0.05)

    sel_noisy = get_knn_acc_all_class(config, data_set, k=config.k, noise_prior=noisy_prior, sel_noisy=sel_noisy,
                                      thre_noise_rate=T_given_noisy, thre_true=T_given_noisy_true)

    sel_noisy = np.array(sel_noisy, dtype=int)
    sel_clean = np.array(list(set(data_set['index'].tolist()) ^ set(sel_noisy)))

    noisy_in_sel_noisy = np.sum(train_dataset['noise_or_not'][sel_noisy]) / sel_noisy.shape[0]
    precision_noisy = noisy_in_sel_noisy
    recall_noisy = np.sum(train_dataset['noise_or_not'][sel_noisy]) / np.sum(train_dataset['noise_or_not'])

    # print(f'[noisy] precision: {precision_noisy}')
    # print(f'[noisy] recall: {recall_noisy}')
    # print(f'[noisy] F1-score: {2.0 * precision_noisy * recall_noisy / (precision_noisy + recall_noisy)}')

    return sel_noisy, sel_clean, data_set['index'], T_given_noisy


def simiFeat(
        num_epochs: int,
        k: int,
        fet: np.ndarray,
        y: np.ndarray,
        method: str
) -> np.ndarray:
    """
    Return a cleaned label set using iterative KNN
    with fuzzy labeling
    """
    assert method in ["vote", "rank"], "Method must be vote or rank"

    global global_dic
    if y.ndim > 1:
        y = np.argmax(y, axis=-1)
    y_clean = y.copy()
    config.num_classes = np.nanmax(y) + 1
    config.cnt = fet.shape[0]
    config.k = k
    if method == 'vote':
        config.method = 'mv'
    else:
        config.method = 'rank1'

    train_dataloader = setup_dataloader(fet, y, shuffle=True)

    train_dataset = {
        'feature': fet[:config.cnt],
        'noisy_label': y[:config.cnt],
        'noise_or_not': np.empty(y.shape[0], dtype=bool),
        'index': np.arange(0, y.shape[0], 1, dtype=int)
        # 'index' : y
    }
    num_training_samples = fet.shape[0]

    sel_noisy_rec = []

    sel_clean_rec = np.zeros((num_epochs, fet.shape[0]))
    sel_times_rec = np.zeros(fet.shape[0])

    record = [[] for _ in range(config.num_classes)]

    wrong_numbers = np.zeros(fet.shape[0])

    for n in range(num_epochs):
        print("Cleaning Epoch: ", n)
        record = [[] for _ in range(config.num_classes)]  # 清空record列表

        for i_batch, (feature, label, index) in enumerate(train_dataloader):
            feature = feature.to(device)
            label = label.to(device, dtype=torch.long)
            for i in range(feature.shape[0]):
                record[label[i]].append({'feature': feature[i].detach().cpu(), 'index': index[i]})
            # if i_batch > 200:
            #     break
            feature = None
            label = None

        sel_noisy, sel_clean, sel_idx, T = noniterate_detection(config, record, train_dataset,
                                                                sel_noisy=sel_noisy_rec.copy())

        if num_epochs > 1:
            sel_clean_rec[n][np.array(sel_clean, dtype=int)] = 1
            sel_times_rec[np.array(sel_idx)] += 1

        if n % 1 == 0:
            # config.method = 'mv'
            aa = np.sum(sel_clean_rec[:n + 1], 0) / sel_times_rec
            nan_flag = np.isnan(aa)
            aa[nan_flag] = 0
            # aa += 0.1

            sel_clean_summary = np.round(aa).astype(bool)
            sel_noisy_summary = np.round(1.0 - aa).astype(bool)
            sel_noisy_summary[nan_flag] = False
            print(
                f'Found {sel_clean_summary.shape[0] - np.sum(sel_clean_summary) - np.sum(nan_flag * 1)} corrupted instances from {sel_clean_summary.shape[0] - np.sum(nan_flag * 1)} instances')

            for i, item_tf in enumerate(sel_clean_summary):
                if item_tf == False:
                    wrong_numbers[i] += 1
        torch.cuda.empty_cache()




    # figure out how to clean y
    # sel_clean is a n x D array where D is # of instances
    # contains each epochs determination of a correctg label
    sel_clean_rec = np.array(sel_clean_rec)
    y_clean = y.copy()
    # for i in range(len(y)):
    #     y_clean[i], _ = stats.mode(sel_clean_rec[: , i], axis=None, keepdims=False)

    # sel_clean_summary == True for clean labels, false otherwise
    # for i, prediction in enumerate(sel_clean_summary):
    #     if not prediction:
    #         # What is the correct label???
    #         y_clean[i] = np.argmax(T[y[i]])
    #         # y_clean[i] = np.round(aa[i])
    # print("I have changed ", np.count_nonzero(sel_clean_summary == False))
    return wrong_numbers


def rising_K_nearest_neighbors(
        X: np.ndarray,
        y: np.ndarray,
        start_k: int,
        end_k: int
):
    """
    All-KNN + HOC
    Predict new labels for noisy instances using rising values of K,
      with a transition matrix estimated using HOC

    Start_k and end_k are both inclusive bounds

    Returns: the cleaned label set
    """
    global global_dic
    if y.ndim > 1:
        y = np.argmax(y, axis=-1)

    num_class = np.nanmax(y) + 1

    y_clean = y.copy()

    config.num_classes = num_class
    config.cnt = X.shape[0]

    change_total = 0

    for k in range(start_k, end_k + 1):
        print(f'Cleaning epoch k={k}')

        config.k = k

        change_for_epoch = 0

        data_set = {
            'feature': torch.Tensor(X),
            'noisy_label': torch.Tensor(y_clean).long(),
        }

        if 'T_init' in global_dic.keys():
            T_init = global_dic['T_init']
        else:
            T_init = None

        if 'p_init' in global_dic.keys():
            p_init = global_dic['p_init']
        else:
            p_init = None

        T, p, global_dic = get_T_global_min_new(
            config, data_set=data_set, max_step=config.max_iter if T_init is None else 20,
            lr=0.1 if T_init is None else 0.01, NumTest=config.G, T0=T_init, p0=p_init,
            global_dic=global_dic
        )

        noisy_prior = np.array([np.count_nonzero(y_clean == i) for i in range(num_class)])
        T_given_noisy = T * p / noisy_prior
        # add randomness
        for i in range(T.shape[0]):
            T_given_noisy[i][i] += np.random.uniform(low=-0.05, high=0.05)

        neigh = NearestNeighbors(n_neighbors=k + 1, radius=1.0, metric='cosine', n_jobs=8)
        neigh.fit(X)
        n = neigh.kneighbors(X, k + 1, return_distance=False)
        n = np.delete(n, 0, axis=1)

        for i, row in enumerate(n):
            neigh_labels = [y_clean[j] for j in row]
            m = stats.mode(neigh_labels, axis=None)
            # m -> tuplle of (mode, count)
            pred_y = m[0][0]
            support = m[1]
            assigned_y = y_clean[i]
            if pred_y != assigned_y:
                y_clean[i] = pred_y if (support * T[assigned_y][pred_y] > (k / 2) or support == k) else assigned_y

        change_for_epoch += np.count_nonzero(y != y_clean) - change_total
        change_total += change_for_epoch
        print(f'Labels changed this epoch : {change_for_epoch}')

    print('Total labels changed: {}'.format(np.count_nonzero(y != y_clean)))
    print(T_given_noisy)
    return y_clean, T_given_noisy

# if __name__ == '__main__':
#     X = np.array([
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0],
#         [0, 1, 0, 1, 1],
#         [1, 0, 0, 1, 1],
#     ])
#
#     y = np.array([1, 0, 0, 1, 1])
#     X_torch = torch.Tensor(X)
#     y_torch = torch.Tensor(y)
#     clstr = compute_apparent_clusterability(X, y)
#     clstr_torch = compute_apparent_clusterability_torch(X_torch, y_torch)
#
#     diff = abs(clstr - clstr_torch)
#     assert diff < 0.01, "diff should be less than 1%"
#     print("Difference of two approaches: ", diff)
#
#     y_clean = simiFeat(10, 3, X, y, "rank")
#     print("y: ", y)
#     print("Y_clean: ", y_clean)
