#!/usr/bin/python
# -*- encoding: utf8 -*-

import optparse
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
import sys
import numpy as np
from sklearn.metrics import roc_curve, f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn import datasets, manifold
import matplotlib.pyplot as plt
from sklearn.linear_model.logistic import LogisticRegression



from sklearn.neighbors import KNeighborsClassifier

__author__ = 'sheep'


def main(node_vec_fname, groundtruth_fname):
    '''\
    %prog [options] <node_vec_fname> <groundtruth_fname>

    groundtruth_fname example: res/karate_club_groups.txt
    '''
    node2vec = load_node2vec(node_vec_fname)
    node2classes = load_node2classes(groundtruth_fname)
    exp_classification(node2classes, node2vec)
    return 0

def load_node2vec(fname):
    node2vec = {}
    with open(fname) as f:
        first = True
        for line in f:
            if first:
                first = False
                continue

            line = line.strip()
            tokens = line.split(' ')
            node2vec[tokens[0]] = map(float, tokens[1:])
    return node2vec

def load_node2classes(fname, is_multiclass=True):
    node2classes = {}
    with open(fname) as f:
        for line in f:
            if line.startswith('#'):
                continue

            node, classes = line.strip().split('\t')
            classes = map(int, classes.split(','))
            if is_multiclass:
                node2classes[node] = classes
            else:
                node2classes[node] = classes[0]
    return node2classes

def exp_classification(node2vec, node2classes, test_mask1, seed=None):

    node2vec = node2vec[test_mask1,:]
    node2classes = node2classes[test_mask1,:]

    classes = set()
    class_list = np.argmax(node2classes, 1)
    print(len(node2vec))
    print(len(node2classes))
    for cs in class_list:
        classes.add(cs)
    print(len(classes))
    X = []
    node2vec = node2vec
    for node_ in node2vec:
        X.append(node_)
    weights = []
    total_scores = []
    for class_ in sorted(classes):
        y = []
        for i in range(len(node2vec)):
            if class_ == class_list[i]:
                y.append(1)
            else:
                y.append(0)

        model = LinearSVC()
        print (class_, sum(y), len(y))
        scores = cross_val_score(model, X, y, cv=10,
                                                  scoring='f1',
                                                  n_jobs=5)
        print ("mean_score"+str(sum(scores)/10))
        total_scores.append(sum(scores)/10)
        weights.append(sum(y))

    print (total_scores)
    print ('macro f1:', sum(total_scores)/len(total_scores))

    micro = 0.0
    for i, s in enumerate(total_scores):
        micro += float(s * weights[i])/sum(weights)
    print ('micro f1:', micro)



def knn_classification(node2vec, node2classes, test_mask1, seed=None):

    node2vec = node2vec[test_mask1,:]
    node2classes = node2classes[test_mask1,:]
    class_list = np.argmax(node2classes, 1)

    print(len(node2vec))
    print(len(node2classes))

    X = []
    node2vec = node2vec
    for node_ in node2vec:
        X.append(node_)

    knn = KNeighborsClassifier(n_neighbors=5)
        # cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集


    f1_micro = cross_val_score(knn, X, class_list, cv=10,
                                              scoring='f1_micro',
                                              n_jobs=5)
    f1_macro = cross_val_score(knn, X, class_list, cv=10,
                                              scoring='f1_macro',
                                              n_jobs=5)
    accuracy = cross_val_score(knn, X, class_list, cv=10,
                               scoring='accuracy',
                               n_jobs=5)
    print ("f1_micro"+str(f1_micro.mean()))
    print ("f1_macro"+str(f1_macro.mean()))
    print ("accuracy"+str(accuracy.mean()))

    # weights.append(sum(y))

    # print (total_scores)
    # print ('macro f1:', sum(total_scores)/len(total_scores))
    #
    # micro = 0.0
    # for i, s in enumerate(total_scores):
    #     micro += float(s * weights[i])/sum(weights)
    # print ('micro f1:', micro)


def KNN(node2vec, node2classes, test_mask1, k=5, split_list=[0.2, 0.4, 0.6, 0.8], time=10, show_train=True, shuffle=True):
    print(test_mask1)
    x = node2vec[test_mask1, :]
    node2classes = node2classes[test_mask1, :]
    y = np.argmax(node2classes, 1)
    print(x.shape)
    print(y.shape)
    average_macro = []
    average_micro = []

    for split in split_list:
        ss = split
        split = int(x.shape[0] * split)
        micro_list = []
        macro_list = []
        if time:
            for i in range(time):
                if shuffle:
                    permutation = np.random.permutation(x.shape[0])
                    x = x[permutation, :]
                    y = y[permutation]
                # x_true = np.array(x_true)
                train_x = x[:split, :]
                test_x = x[split:, :]

                train_y = y[:split]
                test_y = y[split:]

                estimator = KNeighborsClassifier(n_neighbors=k)
                estimator.fit(train_x, train_y)
                y_pred = estimator.predict(test_x)

                f1_macro = f1_score(test_y, y_pred, average='macro')
                f1_micro = f1_score(test_y, y_pred, average='micro')
                macro_list.append(f1_macro)
                micro_list.append(f1_micro)
            print('KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(
                time, ss, k, sum(macro_list) / len(macro_list), sum(micro_list) / len(micro_list)))
            average_macro.append(sum(macro_list) / len(macro_list))
            average_micro.append(sum(micro_list) / len(micro_list))

    print('Softmax(f1_macro: {:.4f}, f1_micro: {:.4f}'.format(
            sum(average_macro) / len(average_macro), sum(average_micro) / len(average_micro)))


def my_Softmax(node2vec, node2classes, test_mask1, split_list=[0.2, 0.4, 0.6, 0.8], time=10, show_train=True, shuffle=True):
    x = node2vec[test_mask1, :]
    node2classes = node2classes[test_mask1, :]
    y = np.argmax(node2classes, 1)
    print(x.shape)
    print(y.shape)
    print(x)
    print(y)
    average_macro = []
    average_micro = []

    for split in split_list:
        ss = split
        split = int(x.shape[0] * split)
        micro_list = []
        macro_list = []
        if time:
            for i in range(time):
                if shuffle:
                    permutation = np.random.permutation(x.shape[0])
                    x = x[permutation, :]
                    y = y[permutation]
                # x_true = np.array(x_true)
                train_x = x[:split, :]
                test_x = x[split:, :]

                train_y = y[:split]
                test_y = y[split:]

                estimator = LogisticRegressionCV(multi_class='multinomial', fit_intercept=True,
                                          penalty='l2', solver='newton-cg')

                estimator.fit(train_x, train_y)
                y_pred = estimator.predict(test_x)
                f1_macro = f1_score(test_y, y_pred, average='macro')
                f1_micro = f1_score(test_y, y_pred, average='micro')
                macro_list.append(f1_macro)
                micro_list.append(f1_micro)
            print('Softmax({}avg, split:{}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(
                time, ss, sum(macro_list) / len(macro_list), sum(micro_list) / len(micro_list)))
            average_macro.append(sum(macro_list) / len(macro_list))
            average_micro.append(sum(micro_list) / len(micro_list))

    print('Softmax(f1_macro: {:.4f}, f1_micro: {:.4f}'.format(
       sum(average_macro) / len(average_macro), sum(average_micro) / len(average_micro)))


def my_Kmeans(node2vec, node2classes, test_mask1, k, time=10, return_NMI=False):
    x = node2vec[test_mask1, :]
    node2classes = node2classes[test_mask1, :]
    y = np.argmax(node2classes, 1)


    estimator = KMeans(n_clusters=k)
    ARI_list = []  # adjusted_rand_score(
    NMI_list = []
    if time:
        # print('KMeans exps {}次 æ±~B平å~]~G '.format(time))
        for i in range(time):
            estimator.fit(x, y)
            y_pred = estimator.predict(x)
            score = normalized_mutual_info_score(y, y_pred)
            NMI_list.append(score)
            s2 = adjusted_rand_score(y, y_pred)
            ARI_list.append(s2)
        # print('NMI_list: {}'.format(NMI_list))
        score = sum(NMI_list) / len(NMI_list)
        s2 = sum(ARI_list) / len(ARI_list)
        print('NMI (10 avg): {:.4f} , ARI (10avg): {:.4f}, k (10 avg): {:.4f} '.format(score, s2, k))

    else:
        estimator.fit(x, y)
        y_pred = estimator.predict(x)
        score = normalized_mutual_info_score(y, y_pred)
        print("NMI on all label data: {:.5f}".format(score))
    if return_NMI:
        return score, s2


def tsne(node2vec, node2classes, test_mask1):
    x = node2vec[test_mask1, :]

    node2classes = node2classes[test_mask1, :]
    y = np.argmax(node2classes, 1)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Y = tsne.fit_transform(x)  # 转换后的输出
    fig, ax = plt.subplots()

    plt.tick_params(labelsize=13)

    plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Spectral)
    # ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    # ax.yaxis.set_major_formatter(NullFormatter())
    # plt.axis('tight')

    plt.show()

    fig.savefig('scatter.eps', dpi=600, format='eps')

if __name__ == '__main__':
    parser = optparse.OptionParser(usage=main.__doc__)
    options, args = parser.parse_args()

    if len(args) != 2:
        parser.print_help()
        sys.exit()

    sys.exit(main(*args))

