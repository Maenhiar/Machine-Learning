import numpy as np
import tensorflow as tf
from NeuralNetwork.FKNNFactory import FKNNFactory
from keras.layers import InputLayer
from PerformaceEvaluation.KFoldCrossValidation import KFoldCrossValidation
from Jacobian.JacobiansComparator import FK_Jacobian, Jacobian_analytical

modelsList = []
modelsPerformances = []

for i in range(3):
    nn2J_2D = FKNNFactory().create2J_2D_NN()
    nn2J_2D.addDenseLayer(64)
    nn2J_2D.addDenseLayer(64)
    nn2J_2D.addDenseLayer(64)
    nn2J_2D.setEarlyStopping(10)
    nn2J_2D.setEpochsNumber(1)
    nn2J_2D.setBatchSize(32)
    nn2J_2D.finalizeAdamModel(0.001)
    modelsList.append(nn2J_2D)

modelPerformances = KFoldCrossValidation().performKFoldCrossValidation(modelsList)
modelsList.clear()
modelsPerformances.append(modelPerformances)

for i in range(3):
    nn2J_2D = FKNNFactory().create2J_2D_NN()
    nn2J_2D.addDenseLayer(64)
    nn2J_2D.addDenseLayer(64)
    nn2J_2D.addDenseLayer(64)
    nn2J_2D.setEarlyStopping(10)
    nn2J_2D.setEpochsNumber(1)
    nn2J_2D.setBatchSize(32)
    nn2J_2D.finalizeAdamModel(0.0001)
    modelsList.append(nn2J_2D)

modelPerformances = KFoldCrossValidation().performKFoldCrossValidation(modelsList)
modelsList.clear()
modelsPerformances.append(modelPerformances)

for i in range(3):
    nn2J_2D = FKNNFactory().create2J_2D_NN()
    nn2J_2D.addDenseLayer(64)
    nn2J_2D.addDenseLayer(64)
    nn2J_2D.addDenseLayer(64)
    nn2J_2D.setEarlyStopping(10)
    nn2J_2D.setEpochsNumber(1)
    nn2J_2D.setBatchSize(32)
    nn2J_2D.finalizeAdamModel(0.003)
    modelsList.append(nn2J_2D)

modelPerformances = KFoldCrossValidation().performKFoldCrossValidation(modelsList)
modelsList.clear()
modelsPerformances.append(modelPerformances)

modelsPerformances.sort(key = lambda model : model.getFinalTrainingSetMSE(), reverse=False)
print("Model performances sorted by best to worst are the following:")

# Itera su ogni oggetto della lista e stampa informazioni sul modello
for obj in modelsPerformances:
    obj.getModel().summary()
    print(obj.getFinalMSE())
    print(obj.getFinalTrainingSetMSE())
    print(f'Tempo di allenamento: {obj.getTrainingTime():.4f} secondi')
    print("\nLayer del modello:")
    for layer in obj.getModel().layers:
        if isinstance(layer, InputLayer):
            # Gestire separatamente il layer di input
            print(f"Layer: {layer.name}, Tipo: {layer.__class__.__name__}, Funzione di attivazione: N/A")
        else:
            # Gli altri layer hanno output_shape
            print(f"Layer: {layer.name}, Tipo: {layer.__class__.__name__}, Funzione di attivazione: {layer.activation.__name__ if layer.activation else 'N/A'}")

    # Informazioni su loss e ottimizzatore
    print("\nInformazioni sul modello:")
    print(f"Funzione di loss utilizzata: {obj.getModel().loss}")
    print(f"Tipo di ottimizzatore utilizzato: {obj.getModel().optimizer.__class__.__name__}")
    
    # Parametri dell'ottimizzatore
    optimizer_config = obj.getModel().optimizer.get_config()  # Ottieni la configurazione dell'ottimizzatore
    print("Parametri dell'ottimizzatore:")
    for param, value in optimizer_config.items():
        print(f"  {param}: {value}")

bestModel = modelsPerformances[0].getModel()

"""nn3J_2D = FKNNFactory().create3J_2D_NN()
nn3J_2D.addDenseLayer(64)
nn3J_2D.addDenseLayer(64)
nn3J_2D.addDenseLayer(64)
nn3J_2D.fitAdam(0.001)

nn5J_3D = FKNNFactory().create5J_3D_NN()
nn5J_3D.addDenseLayer(64)
nn5J_3D.addDenseLayer(64)
nn5J_3D.addDenseLayer(64)
nn5J_3D.fitAdam(0.001)"""

# Calcolare la posizione finale per theta
theta = tf.constant([0.5, 1.0], dtype=tf.float32)  # Angoli di esempio

# Jacobiano computato tramite il modello
jacobian_computed = FK_Jacobian(bestModel, theta)

# Jacobiano analitico
theta_numpy = theta.numpy()  # Converti theta in formato numpy per il calcolo analitico
jacobian_analytical = Jacobian_analytical(theta_numpy)

# Confronta i Jacobiani
print("Jacobiano computato:\n", jacobian_computed.numpy())
print("Jacobiano analitico:\n", jacobian_analytical)
print("Differenza tra i Jacobiani:", np.abs(jacobian_computed.numpy() - jacobian_analytical))