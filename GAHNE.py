# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from GAHNE.ge import DeepWalk
from GAHNE.ge import utils as ut
from GAHNE.utils2 import *
from GAHNE.models import GAHNE
import random
from GAHNE.rewards import *
import scipy.sparse as sp
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score,accuracy_score
import GAHNE.exp_classification as Classifier

import warnings
warnings.filterwarnings("ignore")

import json
# Set random seed
seed = 456
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'DBLP_four_area', 'Dataset string.')  # 'MovieLens', 'Cora', 'DBLP_four_area_back'
flags.DEFINE_string('model', 'GAHNE', 'Model string.')

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('cluster_num', 100, 'Number of epochs to train.')

flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 8, 'Number of units in hidden layer 2.')
# flags.DEFINE_integer('hidden3', 4, 'Number of units in hidden layer 3.')
# flags.DEFINE_integer('Attention', 32, 'Number of units in Attention.')

flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')

dataset_arr = FLAGS.dataset.split('|')
ticks = time.time()
for data_index in range(len(dataset_arr)):
    data_str = dataset_arr[data_index]
    # Load data
    node_objects, features, network_objects, all_y_index, all_y_label, \
    pool_y_index, test_y_index, class_num, all_node_num, new_adj, old_adj = load_data(data_str)
    importance, degree = node_importance_degree(node_objects, old_adj, all_node_num)
    features = sparse_to_tuple(features)
    ##################################
    support = []
    indices = []

    for index in range(len(network_objects)):
        adj = network_objects[index]
        # print(adj)
        indice = list(adj.tocoo().col.reshape(-1)) \
                 + list(adj.tocoo().row.reshape(-1))
        indice = list(set(indice))
        indices.append(indice)

        support.append(preprocess_adj(adj))
    num_supports = len(support)   #support是邻接矩阵

    support_tradition = [preprocess_adj(new_adj)]
    num_supports_tradition = 1

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support_tradition': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports_tradition)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, class_num)),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    }

    # Define model training function
    def GAHNE_train(y_train1, train_mask1, y_val1, val_mask1, y_test1, test_mask1):
        model_t = time.time()
        # Create model
        model = GAHNE(placeholders, input_dim=features[2][1], logging=True, total_num=all_node_num)
        # Initialize session
        sess = tf.Session()
        # Init variables
        sess.run(tf.global_variables_initializer())
        # Train model
        outs_train = []
        val_loss = []
        for epoch in range(FLAGS.epochs):
            feed_dict = construct_feed_dict(features, support, support_tradition, y_train1, train_mask1, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Training step
            outs_train = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs, model.predict(), model.embeddings], feed_dict=feed_dict)
            # Validation
            feed_dict_val = construct_feed_dict(features, support, support_tradition, y_val1, val_mask1, placeholders)
            outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
            val_cost0 = outs_val[0]
            val_acc0 = outs_val[1]

            duration0 = time.time() - model_t
            val_loss.append(val_cost0)
            judge_decrease = False
            for idx in range(len(val_loss)-1):
                if val_loss[idx] > val_loss[idx+1]:
                    judge_decrease = True
                    break
            if judge_decrease == False and len(val_loss) >= 40:
                break
            if len(val_loss) >= 40:
                val_loss = val_loss[1:]
            print("Validation: epoch=" + str(epoch), "  Test set results:", "cost=", "{:.5f}".format(val_cost0),
            "accuracy=", "{:.5f}".format(val_acc0), "time=", "{:.5f}".format(duration0))

        # Testing
        feed_dict_test = construct_feed_dict(features, support, support_tradition, y_test1, test_mask1, placeholders)
        outs_test = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_test)
        outs_micro_macro = sess.run([model.pred, model.actual], feed_dict=feed_dict_test)
        # weights_test1 = sess.run([model.weights()], feed_dict=feed_dict_test)

        test_cost0 = outs_test[0]
        test_acc0 = outs_test[1]
        duration0 = time.time() - model_t
        print("Testing: epoch=" + str(epoch), "  Test set results:", "cost=", "{:.5f}".format(test_cost0),
        "accuracy=", "{:.5f}".format(test_acc0), "time=", "{:.5f}".format(duration0))

        pred = outs_micro_macro[0]
        actual = outs_micro_macro[1]
        print(f1_score(actual, pred, average='macro'))  # 0.26666666666666666
        print(f1_score(actual, pred, average='micro'))  # 0.3333333333333333
        print(accuracy_score(actual, pred))  # 0.26666666666666666
        test_outputs = sess.run([model.embeddings], feed_dict=feed_dict_test)
        print("------------")
        Classifier.KNN(test_outputs[0], y_test1, test_mask1)
        Classifier.my_Kmeans(test_outputs[0], y_test1, test_mask1, k = class_num)
        Classifier.tsne(test_outputs[0], y_test1, test_mask1)
        print("Optimization Finished!")
        #print("weights_test"+str(weights_test1))
        return test_cost0, test_acc0, duration0, outs_train

    pool_y_index = [pool_y_index]
    test_y_index = [test_y_index]
    #  Active Learning
    num_train_nodes = len(pool_y_index[0])
    print(num_train_nodes)
    print(len(test_y_index[0]))
    round_num = len(pool_y_index)
    num_pool_nodes = int(num_train_nodes * 2 / 3)
    print("num_pool_nodes"+str(num_pool_nodes))
    first_batch = int((int)(num_pool_nodes * 1))
    print("total" + str(first_batch))
    # num_pool_nodes = 200
    results = []
    model_times = []
    select_times = []

    for run in range(1):
        result_temp = []
        model_time_temp = []
        select_time_temp = []

        y_all = np.zeros((all_node_num, class_num))
        y_all[all_y_index, :] = all_y_label

        random.shuffle(pool_y_index[run])
        val_idx = pool_y_index[run][num_pool_nodes:num_train_nodes]
        val_mask = sample_mask(val_idx, all_node_num)
        y_val = np.zeros((all_node_num, class_num))
        y_val[val_mask, :] = y_all[val_mask, :]
        pool_idx = pool_y_index[run][0:num_pool_nodes]
        test_idx = test_y_index[run]
        pool_mask = sample_mask(pool_idx, all_node_num)
        test_mask = sample_mask(test_idx, all_node_num)
        y_pool = np.zeros((all_node_num, class_num))
        y_test = np.zeros((all_node_num, class_num))
        y_pool[pool_mask, :] = y_all[pool_mask, :]
        y_test[test_mask, :] = y_all[test_mask, :]
        pool_idx = pool_idx.tolist()
        outs_train = []
        train_idx = []
        idx_select = []
        acc = []
        y_train = np.zeros((all_node_num, class_num))
        idx_select = pool_idx[0:first_batch]
        train_mask = sample_mask(idx_select, all_node_num)
        y_train[train_mask, :] = y_all[train_mask, :]

        pool_idx = list(set(pool_idx) - set(idx_select))
        train_idx = train_idx + idx_select

        print("idx_select" + str(len(idx_select)))
        print("train_length" + str(len(train_idx)))
        train_mask = sample_mask(train_idx, all_node_num)

        test_cost, test_acc, model_duration, outs_train = GAHNE_train(y_train, train_mask, y_val, val_mask, y_test, test_mask)
        print("dataset=" + data_str, " round=" + str(run), "  Test set results:", "cost=",
              "{:.5f}".format(test_cost), "accuracy=", "{:.5f}".format(test_acc), "time=",
              "{:.5f}".format(model_duration), " ratio=", "{:.5f}".format( float(len(train_idx)/(num_train_nodes*2))) )
        acc.append(test_acc)

    print("END")

