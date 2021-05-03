import torch
from torchvision import datasets, models, transforms

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import splitfolders
import time
import copy

def data_labeler(target_dir:str, source_dir:str, bins:int, target_name:str,
                 metadata_path: str = "recipes.csv", sep:str = ";"):
    '''
    Splits raw-data from source_dir into train, val, test with labels based on bins
    :param metadata_path: path with target variable that needs splitted in bins
    :param sep: csv separator
    :param target_dir: target dir for splitted data
    :param source_dir: source dir for raw-data
    :param bins: the # bins to make
    :return: generates the splitted files following PyTorch convention at target_dir
    '''
    # read the metadata and bin the target into deciles using pandas qcut
    meta_data = pd.read_csv(metadata_path, sep=sep)
    meta_data[f'{target_name}_binned'] = meta_data['likes']
    print("bins created")
    # meta_data[f'{target_name}_binned'].value_counts().plot(kind='bar')
    # plt.show()
    labels = [str(i) for i in set(meta_data[f'{target_name}_binned'])]

    # # Assume that the previous time we ran this script the labelled image folder
    # # was created succesfully.
    # if os.path.isdir(target_dir):
    #     return

    # overwrites dir if already existing
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    os.mkdir(target_dir)
    for i in labels:
        os.mkdir(os.path.join(target_dir,str(i)))
    print(f"target directories created at: {target_dir}")
    # Copy each image file to the (per label) output folder
    for source_file in os.listdir(source_dir):
        source_path = os.path.join(source_dir, source_file)
        if os.stat(source_path).st_size == 0:
            print("Warning: File %s is empty. Skipping!" % source_path)
            continue
        pic_id = source_file.split('.')[0]
        label = str(list(meta_data[meta_data['photo_id'] == pic_id][f"{target_name}_binned"])[0])
        shutil.copy(source_path, os.path.join(target_dir, label))
    # split image-folder into train test split to adhere PyTorch structure
    splitfolders.ratio(target_dir, output=f"{target_dir}_splitted", seed=43, ratio=(.8, .1, .1))
    print(f"Splitting and labeling done. Results can be found at: {target_dir}_splitted")


class InstagramDataset:
    '''
    This is our custom dataset class which will load the images, perform transformations on them,
    and load their corresponding labels.
    '''
    # Data augmentation and normalization for training
    # Just normalization for validation and test
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_transforms["test"] = data_transforms["val"]  # equal transformation to test as to val-data

    def __init__(self, img_dir: str, transform: dict = data_transforms, BATCHSIZE: int = 128):
        '''
        initialize instagram dataset
        :param img_dir: directory/root of images, following PyTorch convention of folder structure.
        E.g.:
        -img_dir
        --train
        --test
        --val
        :param transform: a dictionary containing transformations for each subset (train, val, test)
        :param BATCHSIZE: int, the batchsize used, should be power of 2
        '''
        self.img_dir = img_dir
        self.transform = transform
        # load data and apply transformations
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(img_dir, x),
                                                       self.transform[x]) for x in ['train', 'val', 'test']
                               }
        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=BATCHSIZE,
                                                           shuffle=True, num_workers=0)
                            for x in ['train', 'val', 'test']
                            }
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val', 'test']}
        self.class_names = self.image_datasets['train'].classes

    def imshow(self, image_index):
        """Imshow for Tensor."""
        inp = self.image_datasets['train'][image_index][0].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean # denormalize image
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)




class ResNet:

    def __init__(self, dataloaders, dataset_sizes, pretrained: bool = True):
        self.model = models.resnet18(pretrained=pretrained)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes

    def train_model(self, criterion, optimizer, scheduler, num_epochs=25):
        '''
        Function to train the model
        :param criterion: loss criterion
        :param optimizer: optimizer to use
        :param scheduler:
        :param num_epochs: number of epochs (i.e. times that entire dataset is evaluated)
        :return: trained model weights
        '''
        since = time.time()

        model = self.model

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    # convert to float32 to match input. Requirement for MSELoss later.
                    labels = labels.float()
                    # The inputs have 2D shape 128x1, labels now is just 1D 128, make it
                    # 2D 128x1 also.
                    labels = labels.unsqueeze(1)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
#                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(outputs == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def visualize_model(self, model, class_names, num_images=6):
        """
        Visualizes 'num_images' images with prediction
        :param num_images: number of images to predict on
        :return: returns image with prediction
        """
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloaders['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                    InstagramDataset.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)
