import random
import numpy as np
from torch import nn
import torch
import torch.optim as optim
from DatasetLoader.DatasetLoader import DatasetLoader
from CNN.CarRacingCNN import CarRacingCNN
from CNN.NetworkTrainer import NetworkTrainer
from CNN.NetworkTester import NetworkTester 
from Plotters.ChartPlotter import ChartPlotter
from Plotters.ConfusionMatrixPlotter import ConfusionMatrixPlotter
import gymnasium as gym
from Simulator.play_policy_template import play

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_value = 42 
set_seed(seed_value)
torch.use_deterministic_algorithms(True)

batchSize = 64
trainingSetDataLoader = DatasetLoader.getTrainingSetDataLoader(batchSize)
testSetDataLoader = DatasetLoader.getTestSetDataLoader(batchSize)
model = CarRacingCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
epochs = 1

trainingLosses, trainingAccuracies, trainingPrecisions, trainingRecalls, trainingF1Scores, trainingTime = \
    NetworkTrainer.train_loop(trainingSetDataLoader, model, criterion, optimizer, epochs)

ChartPlotter.plot(epochs, trainingLosses, trainingAccuracies, trainingPrecisions, \
                  trainingRecalls, trainingF1Scores)

print("Training time: ", trainingTime)
print('Finished Training')

accuracy, precision, recall, f1Score, classesPrecisions, classesRecalls, \
    classesF1Scores, labels, predictions = NetworkTester.test_loop(testSetDataLoader, model)

print(f"Accuratezza sul test set: {accuracy:.2f}%")
print(f"Precisione media (weighted): {precision:.4f}")
print(f"Recall medio (weighted): {recall:.4f}")
print(f"F1 Score medio (weighted): {f1Score:.4f}")

# Stampa delle metriche per ogni classe
for i in range(5):  # Le tue classi vanno da 0 a 4
    print(f"\nClasse {i}:")
    print(f"Precision: {classesPrecisions[i]:.4f}")
    print(f"Recall: {classesRecalls[i]:.4f}")
    print(f"F1 Score: {classesF1Scores[i]:.4f}")

ConfusionMatrixPlotter.plot(labels, predictions)

print("Done!")

env_arguments = {
    'domain_randomize': False,
    'continuous': False,
    'render_mode': 'human'
}

env_name = 'CarRacing-v3'
env = gym.make(env_name, **env_arguments)

print("Environment:", env_name)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)
play(env, model)