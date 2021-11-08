"""
histogram binning estimation [1] of label posteriors for argmax-selected
predictions linear scaling of the remaining classes' softmax.

[1] Zadrozny, Bianca and Elkan, Charles. Obtaining calibrated probability
    estimates from decision trees and naive bayesian classifiers.
    In ICML, pp. 609–616, 2001.
"""
import pickle
import numpy as np
from scipy.special import softmax

class histogram_binning_posterior_estimator(object):

    def __init__(self,logit,gt,n_bin=15):
        """
        Args:
            logit: ndarray of logit vectors with shape (n_example,n_class)
            gt: ndarray of scaler ground truth for logit vectors with shape (n_example,)
            n_bin: number of bins for histogram binning estimation
        """
        if logit.shape[0]!=gt.shape[0]:
            raise ValueError(f'inconsistent input, numbers of logit and gt are not the same.')

        pred = np.argmax(logit,axis=1)
        self.is_correct = pred==gt
        sm = softmax(logit,axis=1)
        self.sm_argmax = np.array([sm[i,p] for i, p in enumerate(pred)])
        self.n_bin = n_bin

        # get bin precision, i.e., estimated posterior for the bin
        self.bin_precision = -1.0*np.ones([n_bin,],dtype=float)
        self.bin_edges = np.linspace(0,1,n_bin+1)
        bin_lowers = self.bin_edges[:-1]
        bin_uppers = self.bin_edges[1:]

        for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers,bin_uppers)):
            if i < n_bin-1: # left inclusive bin [,)
                in_bin = (self.sm_argmax>=bin_lower) * (self.sm_argmax<bin_upper)
            else: # full inclusive bin [,] for the last bin
                in_bin = (self.sm_argmax>=bin_lower) * (self.sm_argmax<=bin_upper)

            if np.sum(in_bin)==0:
                pass
            else:
                self.bin_precision[i]= np.mean(self.is_correct[in_bin])


    def get_posterior(self,sm_query):
        """
        Args:
            sm_query: scalar, argmax selected class's raw softmax score
        Return:
            res: scalar, estimated label posterior from histogram binning
        """
        if sm_query<0.0 or sm_query>1.0:
            raise ValueError(f'incorrect softmax query input: {sm_query:.2f} exceeding range of (0,1)')
        hist_query,_ = np.histogram([sm_query],self.n_bin,(0,1))
        idx = np.where(hist_query!=0)[0]

        res = self.bin_precision[idx][0]

        # no bin precision available at the query softmax score
        # return its own value back
        if res==-1.0:
            return sm_query
        else:
            return res

    def get_calibrated_softmax_vector(self,sm):
        """
        Args:
            sm: softmax vector of shape (n_class,)
        Return:
            rescaled_sm: calibrated softmax vector of shape (n_class,)
                         with its argmax selected softmax score calibrated
                         according to histogram binning and the remaining
                         softmax scores rescaled linearly such that the
                         softmax scores for all classes sum to 1.0
        """

        pred = np.argmax(sm)
        sm_argmax = sm[pred]
        mask = np.ones_like(sm,dtype=int)
        mask[pred] = 0

        # get calibrated softmax for argmax-selected class
        sm_calib = self.get_posterior(sm_argmax)
        remain_norm = 1.0 - sm_calib
        rescaled_sm = sm*mask
        rescaled_sm = remain_norm*(rescaled_sm/np.sum(rescaled_sm))
        rescaled_sm[pred] = sm_calib
        return rescaled_sm

    def viz_of_mapping_function(self):
        """
        visualize the mapping function estimated by histogram binning
        """
        import matplotlib.pyplot as plt

        # generate query points between 0 and 1
        sm_q = np.linspace(0,1,1000)

        sm_calib = np.empty([len(sm_q),],dtype=float)
        for i, sm in enumerate(sm_q):
            sm_calib[i] = self.get_posterior(sm)

        # plot
        fig = plt.figure()
        plt.plot(sm_q,sm_calib,'-.',label=f'mapping ({self.n_bin} bins)')
        plt.plot(sm_q,sm_q,label='y=x (ideal)')
        plt.xlabel('input (argmax-selected) softmax')
        plt.ylabel('estimated posterior')
        plt.legend()
        plt.title(f'mapping between raw softmax and posterior with histogram binning')
        plt.show()

if __name__=='__main__':

    # setup directories
    import os
    from os.path import join, exists

    dataset_name = 'CIFAR-100' # 'ImageNet-Animal' or 'CIFAR-100'
    home_dir = 'D:\\dataset'
    split_fashion = 'val_test'

    dataset_dir = join(home_dir,dataset_name)
    logit_gt_pairs_dir = join(dataset_dir,split_fashion,'logit_gt_pairs')
    logit_gt_val_path = join(logit_gt_pairs_dir,f'logit_gt_val.pkl')
    logit_gt_test_path = join(logit_gt_pairs_dir,'logit_gt_test.pkl')

    # load validation examples
    with open(logit_gt_val_path,'rb') as f:
        data_ = pickle.load(f)
        logit_val = data_['logit']
        gt_val = data_['gt']
    base_pred_val = np.argmax(logit_val,axis=1)
    acc_val = np.sum(base_pred_val==gt_val)/len(gt_val)
    print(f'load validation examples at {logit_gt_val_path}\n\twith base accuracy: {acc_val:.3f}')

    # load test examples
    with open(logit_gt_test_path,'rb') as f:
        data_ = pickle.load(f)
        logit_test = data_['logit']
        gt_test = data_['gt']
    base_pred_test = np.argmax(logit_test,axis=1)
    acc_test = np.sum(base_pred_test==gt_test)/len(gt_test)
    print(f'load test examples at {logit_gt_test_path}\n\twith base accuracy: {acc_test:.3f}')


    sm_test = softmax(logit_test,axis=1)
    sm_test_argmax = np.array([sm_test[i,base_pred_test[i]] for i in range(len(gt_test))])

    # histogram estimation
    hist_estimator = histogram_binning_posterior_estimator(logit_val,gt_val)

    # visualize the effective mapping function acquired through histogram binning
    hist_estimator.viz_of_mapping_function()

    # get calibrated softmax vectors for test examples
    sm_test_calib = np.empty(sm_test.shape,dtype=np.float32)
    for i in range(sm_test_calib.shape[0]):
        sm_test_calib[i] = hist_estimator.get_calibrated_softmax_vector(sm_test[i])
        print(f'{i+1}/{sm_test_calib.shape[0]} calibrated')



# EOF
