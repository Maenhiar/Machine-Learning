import random
import numpy as np
import torch
from NetworkFitter.NetworkFitter import NetworkFitter
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

print('Training...')
networkFitter = NetworkFitter()
print('Finished Training')

print("Training time: ", networkFitter.getTrainingTime())
trainingLosses, trainingAccuracies, trainingPrecisions, \
                  trainingRecalls, trainingF1Scores = networkFitter.getTrainingMetrics()
ChartPlotter.plot(networkFitter.getEpochsNumber(), trainingLosses, trainingAccuracies, trainingPrecisions, \
                  trainingRecalls, trainingF1Scores)

validationLosses, validationAccuracies, validationPrecisions, \
                  validationRecalls, validationF1Scores = networkFitter.getValidationMetrics()
ChartPlotter.plot(networkFitter.getEpochsNumber(), validationLosses, validationAccuracies, validationPrecisions, \
                  validationRecalls, validationF1Scores)

accuracy, precision, recall, f1Score, classesPrecisions, classesRecalls, \
    classesF1Scores, predictions, labels = networkFitter.getTestMetrics()
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

play(env, networkFitter.getModel())