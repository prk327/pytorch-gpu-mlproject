# Imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
# importing matplotlib modules
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import shutil
import pandas as pd
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import random
from csv import DictWriter
from csv import writer
from csv import QUOTE_MINIMAL
from sklearn.model_selection import train_test_split

# Custom Dataset Class
## Needs ATLEAST 3 class methods
## __init__, __len__, __getitem__

class ImageReader(Dataset):
    '''ImageReader class will read the image and convert it into dataloader compatible class
    
    Args:
        csv_file (string) : The csv file should have two column 1 with Image Name and 2 with Image class as integer
        root_dir (string) : The root directory where the image is saved
        transform (callable, optional) : list of transformation method for feature column
        target_transform (callable, optional) : list of transformation method for target column
        ImagePathCol (string) : The name of the column in csv which represent the image name
        ImageLabelColumn (string) : The name of the column in csv which represent the image label
    '''
    def __init__(self, csv_file,
                 root_dir,
                 transform=None,
                 target_transform=None,
                 ImagePathCol="File Name",
                 ImageLabelColumn="Class Label"):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.ImagePathCol = ImagePathCol
        self.ImageLabelColumn = ImageLabelColumn
        self.img_names = self.df.loc[:,ImagePathCol]
        self.y_labels = self.df.loc[:,ImageLabelColumn]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.y_labels.shape[0]

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.img_names.iloc[index])
        image = mpimg.imread(img_path)
        target = self.y_labels.iloc[index]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

class ImagePreProcessing():
    """
    ImagePreProcessing will arrange the image or files in their respective folder as per their labels
    """
    def __init__(self, csv_file=None, root_dir=None, ImagePathCol=None, ImageLabelColumn=None):
        if csv_file and ImagePathCol and ImageLabelColumn:
            self.df = pd.read_csv(csv_file)
            self.ImagePathCol = ImagePathCol
            self.img_names = self.df.loc[:,ImagePathCol]
            self.ImageLabelColumn = ImageLabelColumn
            self.y_labels = self.df.loc[:,ImageLabelColumn]
        elif csv_file and not ImagePathCol:
            return "csv file must accompany image column and image path"
        # this will help to let decide which train test logic to use
        if csv_file:
            self._supervised=True
        else:
            self._supervised=False
        if not ImagePathCol:
            self.ImagePathCol = "File Name"
        if not ImageLabelColumn:
            self.ImageLabelColumn = "Class Label"
        self.root_dir = root_dir
        # directories where the splitted dataset will lie
        self.train_img_dir = os.path.join(os.path.dirname(self.root_dir), 'SplitData/train')
        self.test_img_dir = os.path.join(os.path.dirname(self.root_dir), 'SplitData/test')
        self.validation_img_dir = os.path.join(os.path.dirname(self.root_dir), 'SplitData/validation')
        self.csv_file_path = os.path.join(os.path.dirname(self.root_dir), 'SplitData')
        self._train_csv_path = self.csv_file_path + '/train.csv'
        self._test_csv_path = self.csv_file_path + '/test.csv'
        self._validation_csv_path = self.csv_file_path + '/validation.csv'
        
    def _shuffleList(self, listObj):
        """
        helper function to shuffle the files
        """
        # using random.sample()
        # to shuffle a list
        res = random.sample(listObj, len(listObj))

        # return shuffled list
        return res
    
    def _append_dict_as_row(self, file_name, dict_of_elem, field_names):
        # Open file in append mode
        with open(file_name, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            dict_writer = DictWriter(write_obj, fieldnames=field_names)
            # Add dictionary as wor in the csv
            dict_writer.writerow(dict_of_elem)

    def _write_new_csv(self, file_name):
        with open(file_name, 'w') as csvfile:
            filewriter = writer(csvfile, delimiter=',', quotechar='|', quoting=QUOTE_MINIMAL)
            filewriter.writerow([self.ImagePathCol, self.ImageLabelColumn])

    def split_dataset_into_3(self, train_ratio, valid_ratio):
        """
        split the dataset in the given path into three subsets(test,validation,train)
        :param train_ratio:
        :param valid_ratio:
        :return:
        """
        _, sub_dirs, _ = next(iter(os.walk(self.root_dir)))  # retrieve name of subdirectories
        sub_dir_item_cnt = [0 for i in range(len(sub_dirs))]  # list for counting items in each sub directory(class)
        
        
        for i, sub_dir in enumerate(sub_dirs):

            # variables to save the sub directory name(class name) and to count the images of each sub directory(class)
            class_name = sub_dir
            sub_dir = os.path.join(self.root_dir, sub_dir)
            sub_dir_item_cnt[i] = len([entry for entry in os.listdir(sub_dir) if os.path.isfile(os.path.join(sub_dir, entry))])

            Orderditems = [entry for entry in os.listdir(sub_dir) if os.path.isfile(os.path.join(sub_dir, entry))]
            items = self._shuffleList(Orderditems)

            # transfer data to trainset
            for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio)):
                if not os.path.exists(self.train_img_dir):
                    os.makedirs(self.train_img_dir)
                    self._write_new_csv(self._train_csv_path)

                source_file = os.path.join(sub_dir, items[item_idx])
                dst_file = os.path.join(self.train_img_dir, items[item_idx])

                field_names = [self.ImagePathCol, self.ImageLabelColumn]
                row_dict = {self.ImagePathCol: items[item_idx], self.ImageLabelColumn: class_name}

                if not os.path.isfile(dst_file):
                    shutil.copyfile(source_file, dst_file)
                    self._append_dict_as_row(self._train_csv_path, row_dict, field_names)

            # transfer data to validation
            for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio) + 1,
                                  round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio))):
                if not os.path.exists(self.validation_img_dir):
                    os.makedirs(self.validation_img_dir)
                    self._write_new_csv(self._validation_csv_path)

                source_file = os.path.join(sub_dir, items[item_idx])
                dst_file = os.path.join(self.validation_img_dir, items[item_idx])

                field_names = [self.ImagePathCol, self.ImageLabelColumn]
                row_dict = {self.ImagePathCol: items[item_idx], self.ImageLabelColumn: class_name}

                if not os.path.isfile(dst_file):
                    shutil.copyfile(source_file, dst_file)
                    self._append_dict_as_row(self._validation_csv_path, row_dict, field_names)

            # transfer data to testset
            for item_idx in range(round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio)) + 1, sub_dir_item_cnt[i]):
                if not os.path.exists(self.test_img_dir):
                    os.makedirs(self.test_img_dir)
                    self._write_new_csv(self._test_csv_path)

                source_file = os.path.join(sub_dir, items[item_idx])
                dst_file = os.path.join(self.test_img_dir, items[item_idx])

                field_names = [self.ImagePathCol, self.ImageLabelColumn]
                row_dict = {self.ImagePathCol: items[item_idx], self.ImageLabelColumn: class_name}

                if not os.path.isfile(dst_file):
                    shutil.copyfile(source_file, dst_file)
                    self._append_dict_as_row(self._test_csv_path, row_dict, field_names)

        return
    
    def _DistributeClassImages(self):
        '''
        DistributeClassImages will create seperate folder under the root directory
        with the label or class name and will copy the images to the folder as per 
        their class/label.
        Note: The csv with the file name and label must be provided at class initialization.
        '''
        Folder = {}
        Label = self.y_labels.unique()
        for i in Label:
            Folder[i] = os.path.join(os.path.dirname(self.root_dir), str(i))
            if not os.path.exists(Folder[i]):
                os.makedirs(Folder[i])
            filename = list(self.df.loc[self.df[self.ImageLabelColumn] == i, self.ImagePathCol])
            for item in filename:
                source_file = os.path.join(self.root_dir, item)
                dst_file = os.path.join(Folder[i], item)
                shutil.copyfile(source_file, dst_file)
            print("The Images are saved into folder: {} as per their label {}".format(Folder[i], i))
            
    def _TrainTestCSV(self, train_ratio, valid_ratio, randomstate):    
        '''
        TrainAndTest Split in csv files
        Note: The csv with the file name and label must be provided at class initialization.
        '''
        testsize = (1-train_ratio)
        X_train, X_test, y_train, y_test = train_test_split(self.df.loc[:,self.ImagePathCol], self.df.loc[:,self.ImageLabelColumn], test_size=testsize, random_state=randomstate)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=valid_ratio, random_state=randomstate)
        
         # transfer the label and file name to train csv
        for filename, label in zip(X_train, y_train):
            if not os.path.exists(self.train_img_dir):
                os.makedirs(self.train_img_dir)
                self._write_new_csv(self._train_csv_path)

            field_names = [self.ImagePathCol, self.ImageLabelColumn]
            row_dict = {self.ImagePathCol: filename, self.ImageLabelColumn: label}

            self._append_dict_as_row(self._train_csv_path, row_dict, field_names)
            
         # transfer the label and file name to test csv
        for filename, label in zip(X_test, y_test):
            if not os.path.exists(self.test_img_dir):
                os.makedirs(self.test_img_dir)
                self._write_new_csv(self._test_csv_path)

            field_names = [self.ImagePathCol, self.ImageLabelColumn]
            row_dict = {self.ImagePathCol: filename, self.ImageLabelColumn: label}

            self._append_dict_as_row(self._test_csv_path, row_dict, field_names)
            
        # transfer the label and file name to validation csv
        for filename, label in zip(X_val, y_val):
            if not os.path.exists(self.validation_img_dir):
                os.makedirs(self.validation_img_dir)
                self._write_new_csv(self._validation_csv_path)

            field_names = [self.ImagePathCol, self.ImageLabelColumn]
            row_dict = {self.ImagePathCol: filename, self.ImageLabelColumn: label}

            self._append_dict_as_row(self._validation_csv_path, row_dict, field_names)
            
        return "The train, test and validation csv are saved in folder: {}".format(self.csv_file_path)
            
    def PlotImageGrid(cls, DataReader, labelMap, fsize=8, grid=3, cmap="gray"):
        '''
        PlotImageGrid will display the image within the grid provided
        
        Args:
            DataReader (ImageReader) : The ImageReader class after loading the data.
            labelMap (Map) : A python map with the labels.
            fsize : (float, float), default: :rc:`figure.figsize` Width, height in inches.
            grid (integer) : the square matrix of size (n,n)
            cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
                   The Colormap instance or registered colormap name used to map
                   scalar data to colors. This parameter is ignored for RGB(A) data.
        '''
        figure = plt.figure(figsize=(fsize, fsize))
        cols, rows = grid, grid
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(DataReader), size=(1,)).item()
            img, label = DataReader[sample_idx]
            figure.add_subplot(rows, cols, i)
            plt.title(labelMap[int(label)])
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap)
        plt.show()

    def Split_DataSets(self, train_ratio=0.75, valid_ratio=0.10, randomstate=35):
        """
        split the dataset in the given path into three subsets(test,validation,train)
        :param train_ratio:
        :param valid_ratio:
        :param randomstate:
        :return:
        """
        if self._supervised:
            self._TrainTestCSV(train_ratio, valid_ratio, randomstate)
        else:
            self.split_dataset_into_3(train_ratio, valid_ratio)
        return "Split Train, Test and Validation Sets!"
        
    def ReadImageFiles(self, custom_transform=None,
                             custome_label_transform=None,
                             batch=32,
                             drop_lst=True,
                             randomsort=True, # want to shuffle the dataset
                             num_cpu=2):
        train_dataset = ImageReader(csv_file=self._train_csv_path,
                            root_dir=self.root_dir,
                            transform=custom_transform,
                            target_transform=custome_label_transform)
        
        test_dataset = ImageReader(csv_file=self._test_csv_path,
                            root_dir=self.root_dir,
                            transform=custom_transform,
                            target_transform=custome_label_transform)

        val_dataset = ImageReader(csv_file=self._validation_csv_path,
                            root_dir=self.root_dir,
                            transform=custom_transform,
                            target_transform=custome_label_transform)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch,
                                  drop_last=drop_lst,
                                  shuffle=randomsort, # want to shuffle the dataset
                                  num_workers=num_cpu) # number processes/CPUs to use
        
        test_loader = DataLoader(dataset=test_dataset) # number processes/CPUs to use

        val_loader = DataLoader(dataset=val_dataset) # number processes/CPUs to use

        return train_dataset, test_dataset, val_dataset, train_loader, test_loader, val_loader
    
    

if __name__ == '__main__':
    
    # import Custom_Data_Loader as CDL # import the script as a module
    
    # Initializing the Preprocessing Class
#     imagePre = ImagePreProcessing(root_dir='../data/MNIST/') # Use when we have seperate folders for image labels

    
    imagePre = ImagePreProcessing(csv_file='../data/mnist_label.csv',
                                  root_dir='../data/MNIST/',
                                  ImagePathCol="File Name",
                                  ImageLabelColumn="Class Label") # Use when we have csv file with the image labels
    
#     imagePre._DistributeClassImages() # this will generate the seperate folder for image labels for just testing

    # this will split the datasets into train test and validation sets randomly
    # note we still need to check the imbalance data in train and test sets
    imagePre.Split_DataSets(.75, .10, 50)
    
    # create the transformation for preprocessing
    custom_transform = transforms.Compose([transforms.ToTensor()])
    Custome_label_transform = transforms.Compose([int,torch.tensor])
    
    # split the data into train test and validation sets
    train_dataset, test_dataset, val_dataset, train_loader, test_loader, val_loader = imagePre.ReadImageFiles(custom_transform=custom_transform,
                                                                                                              custome_label_transform=Custome_label_transform)
    
    # create a label map for printing the image labels
    labels_map = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
    }
    
    # print the image the for randomly sample
    imagePre.PlotImageGrid(train_dataset, labels_map)