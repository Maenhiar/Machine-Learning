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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_value = 42 
set_seed(seed_value)
torch.use_deterministic_algorithms(True)

model = CarRacingCNN2()
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.01)
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

















import sys

import numpy as np

try:
    import gymnasium as gym
except ModuleNotFoundError:
    print('gymnasium module not found. Try to install with')
    print('pip install gymnasium[box2d]')
    sys.exit(1)


# Model Deployment with Gymnasium (Continuous Actions)
import numpy as np
from gymnasium.wrappers import RecordVideo

def play(env, model, predefined_actions):
    seed = 2000
    obs, _ = env.reset(seed=seed)

    # Drop initial frames with no action
    no_action = predefined_actions[0]  # [0.0, 0.0, 0.0]
    for _ in range(50):
        obs, _, _, _, _ = env.step(no_action)

    done = False
    while not done:
        # Preprocess the observation
        img = preprocess_observation(obs, (1, 96, 96, 3))
        input_tensor = torch.tensor(np.expand_dims(img, axis=0), dtype=torch.float32)
        p = model(input_tensor)  # Shape: (1, 5)
        predicted_class = np.argmax(p)  # Integer 0-4

        # Map the predicted class to a predefined action
        action = predefined_actions.get(predicted_class, predefined_actions[0])  # Array

        # Ensure the action is a float32 NumPy array
        action = np.array(action, dtype=np.float32)

        # Step the environment with the action
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()

def preprocess_observation(obs, target_size):
    from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

    # Convert observation to PIL Image
    img = array_to_img(obs)
    # Resize image
    img = img.resize(target_size)
    # Convert to array and normalize
    img = img_to_array(img) / 255.0
    return img

# Define predefined actions (Continuous)
predefined_actions = {
    0: np.array([0.0, 0.0, 0.0], dtype=np.float32),  # No Action
    1: np.array([-1.0, 0.0, 0.0], dtype=np.float32), # Steer Left
    2: np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Steer Right
    3: np.array([0.0, 1.0, 0.0], dtype=np.float32),  # Accelerate (Gas)
    4: np.array([0.0, 0.0, 1.0], dtype=np.float32),  # Brake
}

# Initialize the environment without 'continuous' parameter
env_arguments = {
    'domain_randomize': False,
    'render_mode': 'rgb_array'
}

env_name = 'CarRacing-v3'
env = gym.make(env_name, **env_arguments)

# Wrap the environment to record videos
video_dir = 'video_recordings'
env = RecordVideo(env, video_dir)

print("Environment:", env_name)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

# Play the game using the trained model
play(env, model, predefined_actions)