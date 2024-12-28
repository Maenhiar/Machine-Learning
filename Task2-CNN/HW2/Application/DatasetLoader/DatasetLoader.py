from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation

class DatasetLoader:
    """
    This static class loads the dataset. In particular, it contains the 
    methods to preprocess the training set, split it into training set and 
    validation set and load the test set.
    """
    @staticmethod
    def getTrainingSetDataLoader(batchSize):
        """
        Loads the training set, augments it with images preprocessing and splits 
        it into 2/3 training and 1/3 validation set.

        :param batchSize: The batch sizes used to split the training and validation sets.

        :returns: The augmented training and validation sets.
        :rtype: DataLoader
        """
        originalTrainingSet = datasets.ImageFolder(root="./DataSet/TrainingSet/", transform = DatasetLoader.__getToTensorTransformation())
        preprocessedTrainingSet = datasets.ImageFolder(root = "./DataSet/TrainingSet/", 
                                                       transform = DatasetLoader.__getImagesPreprocessingTransformations())
        trainingSet = ConcatDataset([originalTrainingSet, preprocessedTrainingSet])

        trainingSet, validationSet = train_test_split(trainingSet, test_size = 1/3, random_state = 8)

        trainingSetDataLoader = DataLoader(trainingSet, batch_size = batchSize, shuffle = True)
        validationSetDataLoader = DataLoader(validationSet, batch_size = batchSize, shuffle = False)
        return trainingSetDataLoader, validationSetDataLoader
    
    @staticmethod
    def getTestSetDataLoader(batchSize):
        """
        Loads the test set.

        :param batchSize: The batch sizes used to split the test set.

        :returns: The test set.
        :rtype: DataLoader
        """
        transform = DatasetLoader.__getToTensorTransformation()
        testSet = datasets.ImageFolder(root = "./DataSet/TestSet/", transform = transform)
        testSetDataLoader = DataLoader(testSet, batch_size = batchSize, shuffle = False)
        return testSetDataLoader
    
    @staticmethod
    def __getImagesPreprocessingTransformations():
        # Creates and returns an object that contains 
        # the preprocessing steps to be applied to images.

        transform = Compose([
            RandomHorizontalFlip(),
            RandomRotation(5),
            transforms.RandomResizedCrop(96, scale = (0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
        ])
        return transform
    
    @staticmethod
    def __getToTensorTransformation():
        # Creates and returns an object that simply contains 
        # the transformation of the image to a tensor.

        transform = Compose([
            transforms.ToTensor(),
        ])
        return transform