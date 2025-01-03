
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation
from sklearn.model_selection import train_test_split

class DatasetLoader:
    """
    This static class loads the dataset. In particular, it contains the 
    methods to preprocess the training set, split it into training set and 
    validation set and load the test set.
    """
    def getTrainingSetDataLoader(self, batchSize):
        """
        Loads the training set, augments it with images preprocessing and splits 
        it into 2/3 training and 1/3 validation set.

        :param batchSize: The batch sizes used to split the training and validation sets.

        :returns: The augmented training and validation sets.
        :rtype: DataLoader
        """
        trainingSet = datasets.ImageFolder(root = "./DataSet/TrainingSet/")

        trainingSet, validationSet = train_test_split(trainingSet, test_size = 1/3, random_state = 8)

        trainingSet = DatasetLoader.CustomDataset(trainingSet, self.__getImagesPreprocessingTransformations())
        validationSet = DatasetLoader.CustomDataset(validationSet, self.__getToTensorTransformation())

        trainingSetDataLoader = DataLoader(trainingSet, batch_size = batchSize, shuffle = True)
        validationSetDataLoader = DataLoader(validationSet, batch_size = batchSize, shuffle = False)
        return trainingSetDataLoader, validationSetDataLoader
    
    def getTestSetDataLoader(self, batchSize):
        """
        Loads the test set.

        :param batchSize: The batch sizes used to split the test set.

        :returns: The test set.
        :rtype: DataLoader
        """
        transform = self.__getToTensorTransformation()
        testSet = datasets.ImageFolder(root = "./DataSet/TestSet/", transform = transform)
        testSetDataLoader = DataLoader(testSet, batch_size = batchSize, shuffle = False)
        return testSetDataLoader
    
    def __getToTensorTransformation(self):
        # Creates and returns an object that contains 
        # the preprocessing steps to be applied to images.

        transform = Compose([
            transforms.ToTensor()
        ])
        return transform
    
    def __getImagesPreprocessingTransformations(self):
        # Creates and returns an object that contains 
        # the preprocessing steps to be applied to images.

        transform = Compose([
            RandomHorizontalFlip(),
            RandomRotation(5),
            transforms.RandomResizedCrop(96, scale = (0.8, 1.0)),
            transforms.ToTensor()
        ])
        return transform
        
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transformation, target_class=4):
            self.dataset = dataset
            self.augmentedTransformation = transformation
            self.target_class = target_class

        def __getitem__(self, index):
            img, label = self.dataset[index]
            img = self.augmentedTransformation(img)
            return img, label

        def __len__ (self):
            return len(self.dataset)