import os
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset, BatchSampler, random_split

from mpd import models, losses, datasets, summaries
from mpd.utils import model_loader, pretrain_helper
from torch_robotics.torch_utils.torch_utils import freeze_torch_model_params


@model_loader
def get_model(model_class=None, checkpoint_path=None,
              freeze_loaded_model=False,
              tensor_args=None,
              **kwargs):

    if checkpoint_path is not None:
        model = torch.load(checkpoint_path)
        if freeze_loaded_model:
            freeze_torch_model_params(model)
    else:
        ModelClass = getattr(models, model_class)
        model = ModelClass(**kwargs).to(tensor_args['device'])

    return model


# @model_loader
# def get_model(model_class=None, marginal_prob_sigma=None, device=None, checkpoint_path=None, submodules=None,
#               **kwargs):
#     if marginal_prob_sigma is not None:
#         marginal_prob = MarginalProb(sigma=marginal_prob_sigma)
#         kwargs['marginal_prob_get_std'] = marginal_prob.get_std_fn
#
#     if submodules is not None:
#         for key, value in submodules.items():
#             kwargs[key] = get_model(**value)
#     Model = getattr(models, model_class)
#     model = Model(**kwargs).to(device)
#
#     if checkpoint_path is not None:
#         model.load_state_dict(torch.load(checkpoint_path))
#     if "pretrained_dir" in kwargs and kwargs["pretrained_dir"] is not None:
#         for param in model.parameters():
#             param.requires_grad = False
#     return model

@pretrain_helper
def get_pretrain_model(model_class=None, device=None, checkpoint_path=None, **kwargs):
    Model = getattr(models, model_class)
    model = Model(**kwargs).to(device)

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    return model


def build_module(model_class=None, submodules=None, **kwargs):
    if submodules is not None:
        for key, value in submodules.items():
            kwargs[key] = build_module(**value)

    Model = getattr(models, model_class)
    model = Model(**kwargs)

    return model


def get_loss(loss_class=None, **kwargs):
    LossClass = getattr(losses, loss_class)
    loss = LossClass(**kwargs)
    loss_fn = loss.loss_fn
    return loss_fn

# class CombinedDataset(torch.utils.data.Dataset):
#     def __init__(self, subset_1, subset_2):
#         self.subset_1 = subset_1
#         self.subset_2 = subset_2

#     def __len__(self):
#         # The two subsets should have the same length
#         return len(self.subset_1) + len(self.subset_2)

#     def __getitem__(self, index):
#         # Get the corresponding data from both subsets
#         data_1 = self.subset_1[index]
#         data_2 = self.subset_2[index]
#         return data_1, data_2

# Custom batch sampler
# Custom BatchSampler to achieve the desired batching
# class CustomConcatBatchSampler(BatchSampler):
#     def __init__(self, idx1, idx2, idx3, idx4, n1, n2, n3, n4, batch_size):
#         self.n1 = n1  
#         self.n2 = n2
#         self.n3 = n3
#         self.n4 = n4
#         self.idx1 = idx1
#         self.idx2 = idx2
#         self.idx3 = idx3
#         self.idx4 = idx4
#         self.batch_size = batch_size

#     def __iter__(self):
#         # Iterate through the dataset indices
#         # n1: size of dataset1 and dataset2
#         # n2: size of dataset3 and dataset4
#         i1 = 0 # = 0
#         i2 = 0 # = 16000  
#         i3 = 0 # = 32000
#         i4 = 0 # = 352000

#         while self.idx1 + i1 < self.idx1 + np.floor((self.n1-self.idx1)/(self.batch_size// 2))*(self.batch_size// 2):
#             # print(f'self.idx2--{self.idx2}')
#             # if self.idx1 + i1 < self.idx1 + np.floor((self.n1-self.idx1)/(self.batch_size// 2))*(self.batch_size// 2):
#             batch1 = list(range(self.idx1+i1, min(self.idx1 + i1 + self.batch_size // 2, self.n1)))
#             # print(f'batch1 -- {batch1}')
#             batch2 = list(range(self.idx2+i2,  min(self.idx2 + i2 + self.batch_size // 2, self.n2)))
#             # print(f'batch2 -- {batch2}')
#             yield batch1 + batch2
#             i1 += self.batch_size // 2
#             i2 += self.batch_size // 2
#             # if idx1 >= np.floor(self.n1/self.batch_size)*self.batch_size:
#             #     break

#         while self.idx3 + i3 < self.idx3 + np.floor((self.n3-self.idx3)/(self.batch_size// 2))*(self.batch_size// 2):
#             # if self.idx3 + i3 < self.idx3 + np.floor((self.n3-self.idx3)/(self.batch_size// 2))*(self.batch_size// 2):
#             batch3 = list(range(self.idx3+i3,  min(self.idx3 + i3 + self.batch_size // 2, self.n3)))
#             batch4 = list(range(self.idx4+i4,  min(self.idx4 + i4 + self.batch_size // 2, self.n4)))
#             yield batch3 + batch4
#             i3 += self.batch_size // 2
#             i4 += self.batch_size // 2
#             # if idx2 >= np.floor(self.n2/self.batch_size)*self.batch_size:
#             #     break

#     def __len__(self):
#         return (self.n1 - self.idx1) + (self.n2 - self.idx2) + (self.n3 - self.idx3) + (self.n4 - self.idx4) // (self.batch_size)

def get_specified_dataset(dataset_class=None,
                dataset_subdir=None,
                batch_size=2,
                val_set_size=0.05,
                results_dir=None,
                save_indices=False,
                normal_pos_range = None,
                normal_neg_range = None,
                noisy_pos_range = None,
                noisy_neg_range = None,
                **kwargs):
    DatasetClass = getattr(datasets, dataset_class)
    print('\n---------------Loading data')
    full_dataset = DatasetClass(dataset_subdir=dataset_subdir, **kwargs)
    # data split
    indices_normal_pos = list(normal_pos_range)
    indices_normal_neg = list(normal_neg_range)
    indices_noisy_pos = list(noisy_pos_range)
    indices_noisy_neg = list(noisy_neg_range)

    # train and validation subset
    tran_pos_normal_data_len = tran_neg_normal_data_len  = int((1-val_set_size)*len(indices_normal_pos))

    tran_pos_normal_data_indicies = indices_normal_pos[:tran_pos_normal_data_len]  # pos normal 0,1,... (15200)
    vali_pos_normal_data_indicies = indices_normal_pos[tran_pos_normal_data_len:]  # val: pos normal 15200, 15201,... (800)

    tran_neg_normal_data_indicies = indices_normal_neg[:tran_neg_normal_data_len]  # neg normal 16000,16001,... (15200)
    vali_neg_normal_data_indicies = indices_normal_neg[tran_neg_normal_data_len:]  # val: neg normal 31200, 31201,... (800)

    tran_pos_noisy_data_len = tran_neg_noisy_data_len  = int((1-val_set_size)*len(indices_noisy_pos))

    tran_pos_noisy_data_indicies = indices_noisy_pos[:tran_pos_noisy_data_len]  # pos noisy 32000, 32001, ... (304000)
    vali_pos_noisy_data_indicies = indices_noisy_pos[tran_pos_noisy_data_len:]  # val: pos noisy 336000, 336001,... (16000)

    tran_neg_noisy_data_indicies = indices_noisy_neg[:tran_neg_noisy_data_len]  # neg noisy 352000, 352001, ... (304000)
    vali_neg_noisy_data_indicies = indices_noisy_neg[tran_neg_noisy_data_len:]  # val: neg noisy 656000, 656001,... (16000)

    # Interleave the two lists (train and validation)
    train_normal_combined_indices = [val for pair in zip(tran_pos_normal_data_indicies, tran_neg_normal_data_indicies) for val in pair]
    train_noisy_combined_indices = [val for pair in zip(tran_pos_noisy_data_indicies, tran_neg_noisy_data_indicies) for val in pair]
    train_indices = train_normal_combined_indices + train_noisy_combined_indices

    val_normal_combined_indices = [val for pair in zip(vali_pos_normal_data_indicies, vali_neg_normal_data_indicies) for val in pair]
    val_noisy_combined_indices = [val for pair in zip(vali_pos_noisy_data_indicies, vali_neg_noisy_data_indicies) for val in pair]
    validation_indices = val_normal_combined_indices + val_noisy_combined_indices

    # train subset and validation subset
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, validation_indices)

    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
    print(f'train_dataloader -- {len(train_dataloader)}')

    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    print(f'val_dataloader -- {len(val_dataloader)}')

    if save_indices:
        # save the indices of training and validation sets (for later evaluation)
        torch.save(train_subset.indices, os.path.join(results_dir, f'train_subset_indices.pt'))
        torch.save(val_subset.indices, os.path.join(results_dir, f'val_subset_indices.pt'))

    return train_subset, train_dataloader, val_subset, val_dataloader


def get_dataset(dataset_class=None,
                dataset_subdir=None,
                batch_size=2,
                val_set_size=0.05,
                results_dir=None,
                save_indices=False,
                **kwargs):
    DatasetClass = getattr(datasets, dataset_class)
    print('\n---------------Loading data')
    full_dataset = DatasetClass(dataset_subdir=dataset_subdir, **kwargs)
    print(f'full_subset -- {full_dataset}')
    # print(f'batch_size-- {batch_size}')

    # split into train and validation
    train_subset, val_subset = random_split(full_dataset, [1-val_set_size, val_set_size])
    print(f'train_subset -- {train_subset}')
    train_dataloader = DataLoader(train_subset, batch_size=batch_size)
    print(f'train_dataloader -- {len(train_dataloader)}')
    val_dataloader = DataLoader(val_subset, batch_size=batch_size)
    print(f'val_dataloader -- {len(val_dataloader)}')

    if save_indices:
        # save the indices of training and validation sets (for later evaluation)
        torch.save(train_subset.indices, os.path.join(results_dir, f'train_subset_indices.pt'))
        torch.save(val_subset.indices, os.path.join(results_dir, f'val_subset_indices.pt'))

    return train_subset, train_dataloader, val_subset, val_dataloader


def get_summary(summary_class=None, **kwargs):
    if summary_class is None:
        return None
    SummaryClass = getattr(summaries, summary_class)
    summary_fn = SummaryClass(**kwargs).summary_fn
    return summary_fn


# def get_sampler(sampler_class=None, **kwargs):
#     diffusion_coeff = DiffusionCoefficient(sigma=marginal_prob_sigma)
#     Sampler = getattr(samplers, sampler_class)
#     sampler = Sampler(marginal_prob_get_std_fn=marginal_prob.get_std_fn,
#                       diffusion_coeff_fn=diffusion_coeff,
#                       sde_sigma=marginal_prob_sigma,
#                       **kwargs)
#     return sampler
