import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class Simulator:
    """
    This class executes the simulation of Car Racing with
    the provided model. A video is saved in the project directory
    at the end of the simulation.
    """
    @staticmethod
    def preprocess_observation(obs, target_size):
        img = Image.fromarray(obs)

        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        img_tensor = transform(img)
        return img_tensor

    @staticmethod
    def play(env, model, predefined_actions):
        seed = 2000
        obs, _ = env.reset(seed=seed)

        no_action = predefined_actions[0]
        for _ in range(50):
            obs, _, _, _, _ = env.step(no_action)

        done = False
        while not done:
            img = Simulator.preprocess_observation(obs, target_size=(96, 96))
            device = torch.device("cpu")
            input_tensor = torch.unsqueeze(img, 0).to(device)

            p = model(input_tensor)
            predicted_class = torch.argmax(p, dim=1).item()

            action = predefined_actions.get(predicted_class, predefined_actions[0])
            action = np.array(action, dtype=np.float32)

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        env.close()