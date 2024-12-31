import random
import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.optim as optim
from CNN.CarRacingCNN1 import CarRacingCNN1
from CNN.CarRacingCNN2 import CarRacingCNN2
from ClassificationNeuralNetwork.NetworkFitter import NetworkFitter
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

model = CarRacingCNN1()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
networkFitter = NetworkFitter()
print('Training the model...')
networkFitter.fit(model, optimizer)
print('Finished Training')
print("Training time: ", networkFitter.getTrainingTime())

# Get training performances
trainingLosses, trainingAccuracies, trainingPrecisions, trainingRecalls, \
    trainingF1Scores, trainingLabels, trainingPredictions = networkFitter.getTrainingMetrics()
print("Training results:")
print(classification_report(trainingLabels, trainingPredictions, digits = 3))

# Get validation performances
validationLosses, validationAccuracies, validationPrecisions, validationRecalls, \
    validationF1Scores, validationLabels, validationPredictions = networkFitter.getValidationMetrics()
print("Validation results:")
print(classification_report(validationLabels, validationPredictions, digits = 3))

# Plot training and validation performances
ChartPlotter.plot(networkFitter.getEpochsNumber(), "Loss", trainingLosses, validationLosses)
ChartPlotter.plot(networkFitter.getEpochsNumber(), "Accuracy", trainingAccuracies, validationAccuracies)
ChartPlotter.plot(networkFitter.getEpochsNumber(), "Precision", trainingPrecisions, validationPrecisions)
ChartPlotter.plot(networkFitter.getEpochsNumber(), "Recall", trainingRecalls, validationRecalls)
ChartPlotter.plot(networkFitter.getEpochsNumber(), "F1-Score", trainingF1Scores, validationF1Scores)

# Get test performances
testLoss, _, _, _, _, testLabels, testPredictions = networkFitter.getTestMetrics()
print(testLoss)
print("Test results:")
print(classification_report(testLabels, testPredictions, digits = 3))

# Plot confusion matrix
ConfusionMatrixPlotter.plot(testLabels, testPredictions)

print("Done!")
"""
env_arguments = {
    'domain_randomize': False,
    'continuous': False,
    'render_mode': 'human'
}

env_name = 'CarRacing-v2'
env = gym.make(env_name, **env_arguments)

print("Environment:", env_name)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

model = networkFitter.getTrainedModel()

play(env, model)"""