import numpy as np
import copy as cp


class Normalization:
    ##Normalize the dataset.
    ##The format to be normalized is defined by norm_format.
    ##Run run() to normalize the dataset.
    ##Run inv_run() to restore the normalized dataset.
    def __init__(self, m_list, norm_format, bias, eps):
        self.m_list = m_list
        self.norm_format = norm_format
        self.bias = bias
        self.eps = eps
        print("m_list : {}".format(self.m_list))
        print("norm_format : {}".format(self.norm_format))
        print("bias : {}".format(self.bias))
        if self.bias:
            print("eps : {}".format(self.eps))
        self.param_min, self.param_max = None, None
        self.mean, self.stddev = None, None
        
    def run(self, dataset):
        if self.norm_format == "Normalization":
            dataset_normed = self.__normalize(cp.deepcopy(dataset))
        if self.norm_format == "Standardization":
            dataset_normed = self.__standardization(cp.deepcopy(dataset))
        if self.norm_format == "None":
            dataset_normed = self.__none(cp.deepcopy(dataset))
        return dataset_normed
    
    def inv_run(self, dataset_normed):
        if self.norm_format == "Normalization":
            dataset_inv = self.__inv_normalize(cp.deepcopy(dataset_normed))
        if self.norm_format == "Standardization":
            dataset_inv = self.__standardization(cp.deepcopy(dataset_normed))
        if self.norm_format == "None":
            dataset_inv = self.__none(cp.deepcopy(dataset_normed))
        return dataset_inv
        
    def __normalize(self, dataset):
        param_min, param_max = {}, {}
        for m_key in self.m_list:
            param_min[m_key], param_max[m_key] = np.inf, -np.inf
            for param in dataset[m_key]:
                if param.min() < param_min[m_key]:
                    param_min[m_key] = param.min()
                if param.max() > param_max[m_key]:
                    param_max[m_key] = param.max()
            for param in dataset[m_key]:
                param -= param_min[m_key]
                param /= (param_max[m_key] - param_min[m_key])
                if self.bias:
                    param += self.eps
        self.param_min, self.param_max = param_min, param_max
        return dataset
    
    def __inv_normalize(self, dataset):
        for m_key in self.m_list:
            for param in dataset[m_key]:
                if self.bias:
                    param -= self.eps
                param *= (self.param_max[m_key] - self.param_min[m_key])
                param += self.param_min[m_key]
        return dataset
    
    def __standardize(self, dataset):
        mean, stddev = {}, {}
        for m_key in self.m_list:
            denominator = 0
            mean[m_key] = 0
            stddev[m_key] = 0
            for param in dataset[m_key]:
                mean[m_key] += np.sum(param)
                denominator += param.size()
            for param in dataset[m_key]:
                stddev[m_key] += np.sum((param - mean[m_key])**2)
            stddev[m_key] = np.sqrt(stddev[m_key] / denominator)
            for param in dataset[m_key]:
                param = (param - mean[m_key]) / stddev[m_key]
        self.mean, self.stddev = mean, stddev
        return dataset
    
    def __inv_standardize(self, dataset):
        for m_key in self.m_list:
            for param in dataset[m_key]:
                param *= self.stddev[m_key]
                param += self.mean[m_key]
        return dataset
    
    def __none(self, dataset):
        if self.bias:
            for m_key in self.m_list:
                for param in dataset[m_key]:
                    param += self.eps
        return dataset
    
    def __inv_none(self, dataset):
        if self.bias:
            for m_key in self.m_list:
                for param in dataset[m_key]:
                    param -= self.eps
        return dataset
