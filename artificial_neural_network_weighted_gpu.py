from artificial_neural_network_gpu import ArtificialNeuralNetwork
from multilayer_extend_gpu import MultiLayerNetExtend
from reshape_merger_tree import ReshapeMergerTree
from optimizer_gpu import set_optimizer
import matplotlib.pyplot as plt
from relative_error import relative_error
import numpy as np
import cupy
import copy as cp
import os, sys, shutil


class ArtificialNeuralNetworkWeighted(ArtificialNeuralNetwork):
    ##This class inherits ArtificialNeuralNetwork(in artificiar_neural_network_gpu.py).
    ##Override the learning() to weight a training dataset.
    ##Heavier the self.data_weight of the training dataset, easier the data to be selected as a mini-batch.
    ##The self.data_weight is calculated by relative_error() every 10 epochs. 
    def __init__(self, input_size, hidden, act_func, weight_init, batch_norm, output_size, lastlayer_identity, loss_func, is_epoch_in_each_mlist = False):
        super().__init__(input_size, hidden, act_func, weight_init, batch_norm, output_size, lastlayer_identity, loss_func, is_epoch_in_each_mlist = False)
        self.data_weight = None

    def learning(self, train, test, opt, lr, batchsize_denominator, epoch, m_list):
        ##Initialize the self-variables.
        if self.is_epoch_in_each_mlist:
            for m_key in m_list:
                self.loss_val[m_key] = []
                self.train_acc[m_key], self.test_acc[m_key] = [], []
        ##Make input/output dataset.
        RMT_train, RMT_test = {}, {}
        train_input, train_output = {}, {}
        test_input, test_output = {}, {}
        train_input_, train_output_ = None, None
        test_input_, test_output_ = None, None
        for m_key in m_list:
            RMT_train[m_key] = ReshapeMergerTree()
            RMT_test[m_key] = ReshapeMergerTree()
            train_input[m_key], train_output[m_key] = RMT_train[m_key].make_dataset(train[m_key], self.input_size, self.output_size)
            test_input[m_key], test_output[m_key] = RMT_test[m_key].make_dataset(test[m_key], self.input_size, self.output_size)
            if self.is_epoch_in_each_mlist:
                train_mask = (train_output[m_key] == 0.0)
                test_mask = (test_output[m_key] == 0.0)
                train_output[m_key][train_mask] += 1e-7
                test_output[m_key][test_mask] += 1e-7
            if train_input_ is None:
                train_input_, train_output_ = cp.deepcopy(train_input[m_key]), cp.deepcopy(train_output[m_key])
                test_input_, test_output_ = cp.deepcopy(test_input[m_key]), cp.deepcopy(test_output[m_key])
            else:
                train_input_, train_output_ = np.concatenate([train_input_, train_input[m_key]], axis = 0), np.concatenate([train_output_, train_output[m_key]], axis = 0)
                test_input_, test_output_ = np.concatenate([test_input_, test_input[m_key]], axis = 0), np.concatenate([test_output_, test_output[m_key]], axis = 0)
        train_mask = (train_output_ == 0.0)
        test_mask = (test_output_ == 0.0)
        train_output_[train_mask] += 1e-7
        test_output_[test_mask] += 1e-7
        print("Make a train/test dataset.")
        print("Train dataset size : {}\nTest dataset size : {}".format(train_input_.shape[0], test_input_.shape[0]))
        ##Define the optimizer.
        learning_rate = float(lr)
        optimizer = set_optimizer(opt, learning_rate)
        ##Define the number of iterations.
        rowsize_train = train_input_.shape[0]
        batch_mask_arange = np.arange(rowsize_train)
        batch_size = int(rowsize_train/batchsize_denominator)
        iter_per_epoch = int(rowsize_train/batch_size)
        iter_num = iter_per_epoch * epoch
        self.data_weight = np.ones(train_input_.shape[0]) / train_input_.shape[0]
        print("Mini-batch size : {}\nIterations per 1epoch : {}\nIterations : {}".format(batch_size, iter_per_epoch, iter_num))
        ##Start learning.
        for i in range(iter_num):
            ##Make a mini batch.
            batch_mask = np.random.choice(batch_mask_arange, batch_size, p = self.data_weight)
            batch_input, batch_output = cupy.asarray(train_input_[batch_mask, :]), cupy.asarray(train_output_[batch_mask, :])
            ##Update the self.network.params with grads.
            grads = self.network.gradient(batch_input, batch_output, is_training = True)
            params_network = self.network.params
            optimizer.update(params_network, grads, i)
            ##When the iteration i reaches a multiple of iter_per_epoch,
            ##Save loss_values, train/test_accuracy_value of the self.network to self.loss_val, self.train_acc, self.test_acc.
            if i % iter_per_epoch == 0:
                if self.is_epoch_in_each_mlist:
                    for m_key in m_list:
                        loss_val = self.network.loss(cupy.asarray(train_input[m_key]), cupy.asarray(train_output[m_key]), is_training = False)
                        self.loss_val[m_key].append(loss_val)
                        train_acc = self.network.accuracy(cupy.asarray(train_input[m_key]), cupy.asarray(train_output[m_key]), is_training = False)
                        self.train_acc[m_key].append(train_acc)
                        test_acc = self.network.accuracy(cupy.asarray(test_input[m_key]), cupy.asarray(test_output[m_key]), is_training = False)
                        self.test_acc[m_key].append(test_acc)
                else:
                    loss_val = self.network.loss(cupy.asarray(train_input_), cupy.asarray(train_output_), is_training = False)
                    self.loss_val.append(loss_val)
                    train_acc = self.network.accuracy(cupy.asarray(train_input_), cupy.asarray(train_output_), is_training = False)
                    self.train_acc.append(train_acc)
                    test_acc = self.network.accuracy(cupy.asarray(test_input_), cupy.asarray(test_output_), is_training = False)
                    self.test_acc.append(test_acc)
            if i != 0 and i % (10 * iter_per_epoch) == 0:
                if self.is_epoch_in_each_mlist:
                    data_weight = None
                    for m_key in m_list:
                        predict = self.network.predict(cupy.asarray(train_input[m_key]), is_training = False)
                        predict = cupy.asnumpy(predict)
                        error = relative_error(train_output[m_key], predict)
                        if data_weight is None:
                           data_weight = error
                        else:
                            data_weight = np.concatenate([data_weight, error], axis = 0)
                    data_weight = np.mean(data_weight, axis = 1)
                    data_weight /= data_weight.sum()
                else:
                    predict = self.network.predict(cupy.asarray(train_input_), is_training = False)
                    predict = cupy.asnumpy(predict)
                    error = relative_error(train_output_, predict)
                    error = np.mean(error, axis = 1)
                    data_weight = error / error.sum()
                self.data_weight = data_weight
