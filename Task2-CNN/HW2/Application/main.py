import torch
from torch import nn
from  DatasetLoader.DatasetLoader import DatasetLoader
from NetworkTrainer.NetowrkTrainer import NetowrkTrainer
from CNN.CNN import CNN

batchSize = 64
trainingSetDataLoader = DatasetLoader.getTrainingSetDataLoader(batchSize)
testSetDataLoader = DatasetLoader.getTestSetDataLoader(batchSize)
model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    NetowrkTrainer.train_loop(trainingSetDataLoader, model, loss_fn, optimizer, batchSize)
    NetowrkTrainer.test_loop(testSetDataLoader, model, loss_fn)
print("Done!")