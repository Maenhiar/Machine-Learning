import torch
from torch import nn
from  DatasetLoader.DatasetLoader import DatasetLoader
from NetworkTrainer.NetowrkTrainer import NetowrkTrainer
from CNN.CNN import CNN
import torch.optim as optim

batchSize = 64
trainingSetDataLoader = DatasetLoader.getTrainingSetDataLoader(batchSize)
testSetDataLoader = DatasetLoader.getTestSetDataLoader(batchSize)
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
epochs = 50

NetowrkTrainer.train_loop(trainingSetDataLoader, model, criterion, optimizer, epochs)
NetowrkTrainer.test_loop(testSetDataLoader, model)

print("Done!")