import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

# Carica i dataset
train_dataset = datasets.ImageFolder(root = "./DataSet/TrainingSet/", transform = transform)
test_dataset = datasets.ImageFolder(root = "./DataSet/TestSet/", transform = transform)

# Crea i DataLoader
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

# Verifica la struttura del dataset
print(f"Numero di classi: {len(train_dataset.classes)}")
print(f"Classi: {train_dataset.classes}")

# Ottieni un batch di immagini e etichette
images, labels = next(iter(train_loader))
print(f"Dimensione del batch di immagini: {images.size()}")
print(f"Etichette: {labels}")
