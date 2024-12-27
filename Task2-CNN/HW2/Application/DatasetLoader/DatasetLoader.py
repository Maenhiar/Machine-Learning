from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation

class DatasetLoader:
    
    @staticmethod
    def getTrainingSetDataLoader(batchSize):
        originalTrainingSet = datasets.ImageFolder(root="./DataSet/TrainingSet/", transform = DatasetLoader.__getToTensorTransform())
        preprocessedTrainingSet = datasets.ImageFolder(root = "./DataSet/TrainingSet/", 
                                                       transform = DatasetLoader.__getImagesPreprocessingTransform())
        trainingSet = ConcatDataset([originalTrainingSet, preprocessedTrainingSet])

        # Split dataset in 2/3 for training set and 1/3 for validation set.
        trainingSet, validationSet = train_test_split(trainingSet, test_size = 1/3, random_state = 8)

        trainingSetDataLoader = DataLoader(trainingSet, batch_size = batchSize, shuffle = True)
        validationSetDataLoader = DataLoader(validationSet, batch_size = batchSize, shuffle = False)
        return trainingSetDataLoader, validationSetDataLoader
    
    @staticmethod
    def getTestSetDataLoader(batchSize):
        transform = DatasetLoader.__getToTensorTransform()
        testSet = datasets.ImageFolder(root = "./DataSet/TestSet/", transform = transform)
        testSetDataLoader = DataLoader(testSet, batch_size = batchSize, shuffle = False)
        return testSetDataLoader
    
    @staticmethod
    def __getImagesPreprocessingTransform():
        transform = Compose([
            RandomHorizontalFlip(),
            RandomRotation(5),
            transforms.RandomResizedCrop(96, scale = (0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
        ])
        return transform
    
    @staticmethod
    def __getToTensorTransform():
        transform = Compose([
            transforms.ToTensor(),
        ])
        return transform