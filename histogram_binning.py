"""
Authors: Tong Liang, Jim Davis
Affiliation: Computer Vision Lab, Ohio State University
Email: liang.693@osu.edu
Date: 11/14/2021

This is an implementation of histogram binning calibration [1] for argmax-selected
predictions, followed by linear scaling of the remaining classes' softmax scores.

[1] Zadrozny, Bianca and Elkan, Charles. Obtaining calibrated probability
    estimates from decision trees and naive bayesian classifiers.
    In ICML, pp. 609–616, 2001.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from torch.nn import functional as F
from models import *

class histogram_binning_calibration(nn.Module):

    def __init__(self, model, n_bins=15, device='cpu'):
        super(histogram_binning_posterior_estimator,self).__init__()
        self.base_model = model.to(device)
        self.base_model.eval()

        self.n_bins = n_bins
        self.device = device

    def forward(self, input):
        """
        applying histogram binning + linear rescaling to input logits
        need to call histogram_binning() method first
        Args:
            input: input examples for the base model, e.g., 1 batch of image
                   tensors of shape (batch_size, channels, height, width)
        Returns:
            calibrated softmax vectors of shape (batch_size, num_classes)
        """

        input = input.to(self.device)
        logits = self.base_model(input)
        softmax_output = F.softmax(logits,dim=1)

        return self.get_calibrated_softmax_vector(softmax_output)


    def histogram_binning(self, val_loader, verbose=False):
        """
        apply histogram binning [1] classifier calibration approach
        to softmax scores of top1 predictions output by the base model
        Args:
            val_loader: dataloader points to validation set
            verbose: set true to print out the progress
        """

        bin_edges = torch.linspace(0, 1, self.n_bins + 1)
        self.bin_lowers = bin_edges[:-1]
        self.bin_uppers = bin_edges[1:]
        # histogram carrying the bin precisions
        self.histogram = -1.0*torch.ones([self.n_bins,])

        # collect logits and labels
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for i, (input, label) in enumerate(val_loader):
                input = input.to(self.device)
                logits = self.base_model(input)
                logits_list.append(logits)
                labels_list.append(label)
                if verbose:
                    print(f'extracting validation logits {i+1}/{len(val_loader)} batches')
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

            # compute the histogram
            softmaxes = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(softmaxes, 1)
            accuracies = predictions.eq(labels)

            for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
                """compute |confidence - accuracy| in each bin"""
                # right inclusive bins (,]
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    self.histogram[i] = accuracies[in_bin].float().mean().squeeze()

                if verbose:
                    print(f'estimating {i+1}/{self.n_bins} bin of the histogram')

    def get_posterior(self,sm_query):
        """
        Args:
            sm_query: torch.tensor(scalar), argmax selected class's raw softmax score
        Return:
            res: tuple(torch.tensor(scalar), flag), where torch.tensor(scalar)
                 is the estimated label posterior from histogram binning, and
                 flag indicates whether the calibration is valid (1) or not (0)
                 Estimated value could be undefined due to encountering of empty
                 bin corresponding to the input sm_query softmax value
        """
        if sm_query<0.0 or sm_query>1.0:
            raise ValueError(f'incorrect softmax query input: {sm_query:.2f} exceeding range of (0,1)')

        hist_query = torch.histc(sm_query, bins=self.n_bins, min=0.0, max=1.0)
        idx = torch.where(hist_query!=0)[0]

        res = self.histogram[idx][0]

        # no bin precision available at the query softmax score
        # return its own value back
        if res == -1.0:
            return (sm_query, 0)
        else:
            return (res, 1)

    def get_calibrated_softmax_vector(self,sm):
        """
        Args:
            sm: softmax vector of shape (n_example, n_class)
        Return:
            sm_calib: calibrated softmax vector of shape (n_example, n_class)
                      with its argmax selected softmax score calibrated
                      according to histogram binning and the remaining
                      softmax scores rescaled linearly such that the
                      softmax scores for all classes sum to 1.0, this may
                      alter the argmax-selection result
            valid_flag: a vector of scaler flags of shape (n_example,) indicating
                        if the histogram binning calibration for the associated
                        example is valid (1) or not (0)
            predictions: predictions derived from argmax selection based on the
                         raw examlpes
        """

        with torch.no_grad():
            # this process a batch of examples at a time
            sm_argmax, predictions = torch.max(sm, dim=1)
            n_examples, n_classes = sm.shape

            sm_calib = torch.empty([n_examples, n_classes], dtype=float)
            valid_flag = torch.empty([n_examples,])
            for i, (sm_pred, pred) in enumerate(zip(sm_argmax,predictions)):
                est_posterior, cnt = self.get_posterior(sm_pred)
                valid_flag[i] = cnt
                mask = torch.ones_like(sm[0], dtype=float)
                mask[pred] = 0
                remain_norm = 1.0 - est_posterior
                rescaled_sm = sm[i]*mask
                rescaled_sm = remain_norm*(rescaled_sm/torch.sum(rescaled_sm))
                rescaled_sm[pred] = est_posterior
                # in case some tiny numerical impreicison
                sm_calib[i] = rescaled_sm/torch.sum(rescaled_sm)

            # print(f'valid calibration: {torch.sum(valid_flag).item()}/{n_examples}')
            return sm_calib, valid_flag, predictions

    def viz_of_mapping_function(self):
        """
        visualize the mapping function estimated by histogram binning
        """
        import matplotlib.pyplot as plt

        # generate query points between 0 and 1
        sm_q = torch.linspace(0,1,1000)

        sm_calib = torch.empty([len(sm_q),],dtype=float)
        for i, sm in enumerate(sm_q):
            sm_calib[i], _ = self.get_posterior(sm)

        # convert to ndarray
        sm_q = sm_q.numpy()
        sm_calib = sm_calib.cpu().numpy()

        # plot
        fig = plt.figure()
        plt.plot(sm_q,sm_calib,'-.',label=f'mapping ({self.n_bins} bins)')
        plt.plot(sm_q,sm_q,label='y=x (ideal)')
        plt.xlabel('input (argmax-selected) softmax')
        plt.ylabel('estimated posterior')
        plt.legend()
        plt.title(f'mapping between raw softmax and posterior with histogram binning')
        plt.show()


if __name__=='__main__':

    # setup directories ------------------------------------------------------ #
    import os
    from os.path import join, exists

    dataset_name = 'iNat2019_full_nat04d' #
    home_dir = 'C:\\DATASET'
    dataset_dir = join(home_dir,dataset_name)
    val_dir = join(dataset_dir,'val')
    test_dir = join(dataset_dir,'test')

    # base model path
    model_path = 'D:\\dataset\\iNat2019_full_nat04d_record\\resnet18_kldivloss_pretrained_baseline\\epoch_273_model.th'

    # setup dataloader ------------------------------------------------------- #
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets

    # normalization for iNat2019
    mean_inat19 = [0.454, 0.474, 0.367]
    std_inat19 = [0.237, 0.230, 0.249]
    normalize = transforms.Normalize(mean=mean_inat19, std=std_inat19)

    eval_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
    ])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root=val_dir,transform=eval_transforms),
        batch_size = 256,
        shuffle = False,
        num_workers = 8,
        pin_memory = False,
        drop_last = False
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root=test_dir,transform=eval_transforms),
        batch_size = 256,
        shuffle = False,
        num_workers = 8,
        pin_memory = False,
        drop_last = False
    )


    # apply histogram binning approach --------------------------------------- #

    # load base model
    base_model = load_model(model_path)
    device = 'cuda'

    # init class instance
    n_bins = 15
    hist_est = histogram_binning_calibration(base_model,n_bins,device)

    # run histogram binning on validation set
    hist_est.histogram_binning(val_loader,True)
    hist = hist_est.histogram.cpu().numpy()
    print(f'learned histogram: \n{hist}')

    # viz of the histogram binning mapping function
    hist_est.viz_of_mapping_function()

    # applying histogram binning + linear rescaling calibration to test set -- #
    sm_list = []
    label_list = []
    flag_list = []
    raw_preds_list = []

    start = time.time()
    with torch.no_grad():
        for i, (input, label) in enumerate(test_loader):
            input = input.to(device)
            # return calibrated softmax vector and flags for valid calibration
            sm_calib, flags, preds = hist_est(input)
            sm_list.append(sm_calib)
            raw_preds_list.append(preds)
            flag_list.append(flags)
            label_list.append(label)
            print(f'calibrating {i+1}/{len(test_loader)} batches')

    end = time.time()
    print(f'elapse time: {end-start:.2f} sec')

    # calibrated softmax vectors and its assocaited ground truth labels
    sm_list = torch.cat(sm_list).cpu().numpy()
    label_list = torch.cat(label_list).cpu().numpy()
    flag_list = torch.cat(flag_list).cpu().numpy()
    raw_preds_list = torch.cat(raw_preds_list).cpu().numpy()

    print(f'valid calibration examples: {int(np.sum(flag_list))}/{len(flag_list)}')

    acc_raw = np.sum(raw_preds_list==label_list)/len(label_list)
    print(f'original model prediction accuracy {acc_raw*100:.2f}%')

    pred = np.argmax(sm_list,axis=1)
    acc_calib = np.sum(pred==label_list)/len(label_list)
    print(f'calibrated model prediction accuracy {acc_calib*100:.2f}%')






# EOF
