from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DatasetLoader:
    
    @staticmethod
    def getTrainingSetDataLoader(batchSize):
        transform = DatasetLoader.__getTransform()        
        trainingSet = datasets.ImageFolder(root = "./DataSet/TrainingSet/", transform = transform)
        trainingSetDataLoader = DataLoader(trainingSet, batch_size = batchSize, shuffle = True)
        
        return trainingSetDataLoader
    
    @staticmethod
    def getTestSetDataLoader(batchSize):
        transform = DatasetLoader.__getTransform()
        testSet = datasets.ImageFolder(root = "./DataSet/TestSet/", transform = transform)
        testSetDataLoader = DataLoader(testSet, batch_size = batchSize, shuffle = False)
        
        return testSetDataLoader
    
    @staticmethod
    def __getTransform():
        return transforms.Compose([transforms.ToTensor()])