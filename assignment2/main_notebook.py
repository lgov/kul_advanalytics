#!/usr/bin/env python
# coding: utf-8

# # Getting started
# * To install PyTorch. Follow the steps in [this](https://pytorch.org/get-started/locally/) link
# * see [here](https://pypi.org/project/split-folders/) for more information on the splitfolders package used in 'helper'
# * general example to follow: see [here](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), [here](https://towardsdatascience.com/a-practical-example-in-transfer-learning-with-pytorch-846bb835f2db), or [here](https://towardsdatascience.com/transfer-learning-with-convolutional-neural-networks-in-pytorch-dd09190245ce)

import os
from multiprocessing.spawn import freeze_support

freeze_support()

# pytorch stuff
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# custom imports
from helper import InstagramDataset, data_labeler, ResNet

# # Labeling
bins = 10
# note that splitting the data into split, val, test and reorganizing folders accordingly is a slow process
# this operation can take up to 15-30min. Check the function to get an idea what is going on in the background
data_labeler(target_dir='recipes_labeled', source_dir='recipes/recipes/', bins=bins,
             target_name='likes', metadata_path='recipes.csv', sep=';')


# # Reading Instadata following PyTorch convention
# here we load the Instagram dataset through a custom class, this class reads the data, applies transformations,
# and creates batches for train, test, and val data as iterators (i.e. dataloaders)
# see docstring for further information
insta_data = InstagramDataset('recipes_labeled_splitted')
dataloaders = insta_data.dataloaders
# # check random image, works especially well if you're hungry... :)
# insta_data.imshow(15)


# # The model
resn = ResNet(dataloaders, insta_data.dataset_sizes, pretrained=True) #initialize the ResNet defined in helper
model = resn.model
model.eval() #the original resnet 50 structure

# Freeze model weights
for param in model.parameters():
    param.requires_grad = False

# extract number of nodes in last fc layer and add own fc layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1) # from 2048 to 10 (i.e. our target)

model = model.to(resn.device)

# set objective criterion
criterion = nn.MSELoss()

# Observe that only params in last fc layer are optimized
optimizer_ft = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=6, gamma=0.1)
# observe how the last layer has been replaced by our own layer
model.eval()
resn.train_model(criterion, optimizer_ft, exp_lr_scheduler, num_epochs=14)





