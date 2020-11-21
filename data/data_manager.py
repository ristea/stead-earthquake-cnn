from data.base_dataset import BaseDataset
import numpy as np
import torch


class DataManager:
    def __init__(self, config):
        self.config = config

    def get_dataloader(self):
        dataset = BaseDataset(self.config)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            pin_memory=True if self.config['device'] == 'cuda' else False
        )
        return dataloader

    def get_train_eval_dataloaders(self):
        np.random.seed(707)

        dataset = BaseDataset(self.config)
        dataset_size = len(dataset)

        ## SPLIT DATASET
        train_split = self.config['train_size']
        train_size = int(train_split * dataset_size)
        validation_size = dataset_size - train_size

        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        temp = int(train_size + validation_size)
        val_indices = indices[train_size:temp]

        ## DATA LOARDER ##
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=self.config['batch_size'],
                                                   sampler=train_sampler,
                                                   pin_memory=True if self.config['device'] == 'cuda' else False)

        validation_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                        batch_size=self.config['batch_size'],
                                                        sampler=valid_sampler,
                                                        pin_memory=True if self.config['device'] == 'cuda' else False)
        return train_loader, validation_loader

    def get_train_eval_test_dataloaders(self):
        np.random.seed(707)

        dataset = BaseDataset(self.config)
        dataset_size = len(dataset)

        ## SPLIT DATASET
        train_split = self.config['train_size']
        valid_split = self.config['valid_size']
        test_split = self.config['test_size']

        train_size = int(train_split * dataset_size)
        valid_size = int(valid_split * dataset_size)
        test_size = dataset_size - train_size - valid_size

        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:(train_size + valid_size)]
        test_indices = indices[(train_size + valid_size):]

        ## DATA LOARDER ##
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=self.config['batch_size'],
                                                   sampler=train_sampler,
                                                   pin_memory=True if self.config['device'] == 'cuda' else False)

        validation_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                        batch_size=self.config['batch_size'],
                                                        sampler=valid_sampler,
                                                        pin_memory=True if self.config['device'] == 'cuda' else False)

        test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=self.config['batch_size'],
                                                  sampler=test_sampler,
                                                  pin_memory=True if self.config['device'] == 'cuda' else False)

        return train_loader, validation_loader, test_loader
