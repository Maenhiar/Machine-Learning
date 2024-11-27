import numpy as np
import tensorflow as tf
from Charts.LossPlotter import LossPlotter
from NeuralNetwork.FKNNFactory import FKNNFactory
from keras.layers import InputLayer
from PerformaceEvaluation.KFoldCrossValidation import KFoldCrossValidation
from Jacobian.JacobiansComparator import FK_Jacobian, computeAnalyticalJacobian


def computeJacobians(bestModel):
    theta = tf.constant([0.5, 1.0], dtype=tf.float32)
    computedJacobian = FK_Jacobian(bestModel, theta)
    numpyTheta = theta.numpy()
    jacobian_analytical = computeAnalyticalJacobian(numpyTheta)
    
    print("Jacobiano analitico:\n", jacobian_analytical)
    print("Jacobiano computato:\n", computedJacobian.numpy())
    print("Differenza tra i Jacobiani:", np.abs(computedJacobian.numpy() - jacobian_analytical))

def printModelDataAndPerformances(model):
    model.getModel().summary()

    print("Final MSE = ", model.getFinalMSE())
    print("Final test ste MSE = ", model.getFinalTestSetMSE())
    print(f'Training time: {model.getTrainingTime():.4f} secondi')
    
    print("\Model layers:")
    for layer in model.getModel().layers:
        if isinstance(layer, InputLayer):
            print(f"Layer: {layer.name}, Tipo: {layer.__class__.__name__}, Activation function: N/A")
        else:
            print(f"Layer: {layer.name}, Tipo: {layer.__class__.__name__}, Activation function: {layer.activation.__name__ if layer.activation else 'N/A'}")

    print(f"Loss function: {model.getModel().loss}")

    optimizer_config = model.getModel().optimizer.get_config()
    print("Optimizer parameters:")
    for param, value in optimizer_config.items():
        print(f"  {param}: {value}")
    
    LossPlotter.plotLoss(model.getModelHistory())

modelsPerformances = []

modelsList = []
for i in range(3):
    nn2J_2D = FKNNFactory().create2J_2D_NN()
    nn2J_2DModel = nn2J_2D["model"]
    nn2J_2DModel.addDenseLayer(64)
    nn2J_2DModel.addDenseLayer(64)
    nn2J_2DModel.addDenseLayer(64)
    nn2J_2DModel.setEarlyStopping(10)
    nn2J_2DModel.setEpochsNumber(1)
    nn2J_2DModel.setBatchSize(32)
    nn2J_2DModel.finalizeAdamModel(0.001)
    modelsList.append(nn2J_2DModel)
    nn2J_2DModel.getModel().summary()

modelPerformances = KFoldCrossValidation().performKFoldCrossValidation(modelsList, nn2J_2D["training-set-input"],
                                                                        nn2J_2D["training-set-output"], nn2J_2D["test-set-input"],
                                                                            nn2J_2D["test-set-output"])
modelsList.clear()

modelsPerformances.append(modelPerformances)
modelsPerformances.sort(key = lambda model : model.getFinalTestSetMSE(), reverse = False)

bestModel = modelsPerformances[0].getModel()
print("Model performances sorted by best to worst are the following:")
for modelPerformances in modelsPerformances:
    printModelDataAndPerformances(modelPerformances)

computeJacobians(bestModel)

modelsPerformances.clear()

#3J-2D
for i in range(3):
    nn3J_2D = FKNNFactory().create3J_2D_NN()
    nn3J_2DModel = nn3J_2D["model"]
    nn3J_2DModel.addDenseLayer(64)
    nn3J_2DModel.addDenseLayer(64)
    nn3J_2DModel.addDenseLayer(64)
    nn3J_2DModel.setEarlyStopping(10)
    nn3J_2DModel.setEpochsNumber(1)
    nn3J_2DModel.setBatchSize(32)
    nn3J_2DModel.finalizeAdamModel(0.001)
    modelsList.append(nn3J_2DModel)
    nn3J_2DModel.getModel().summary()

modelPerformances = KFoldCrossValidation().performKFoldCrossValidation(modelsList, nn3J_2D["training-set-input"],
                                                                        nn3J_2D["training-set-output"], nn3J_2D["test-set-input"],
                                                                            nn3J_2D["test-set-output"])
modelsList.clear()

modelsPerformances.append(modelPerformances)
modelsPerformances.sort(key = lambda model : model.getFinalTestSetMSE(), reverse = False)

bestModel = modelsPerformances[0].getModel()
print("Model performances sorted by best to worst are the following:")
for modelPerformances in modelsPerformances:
    printModelDataAndPerformances(modelPerformances)

modelsPerformances.clear()

#5G-3D
for i in range(3):
    nn5J_3D = FKNNFactory().create5J_3D_NN()
    nn5J_3DModel = nn5J_3D["model"]
    nn5J_3DModel.addDenseLayer(64)
    nn5J_3DModel.addDenseLayer(64)
    nn5J_3DModel.addDenseLayer(64)
    nn5J_3DModel.setEarlyStopping(10)
    nn5J_3DModel.setEpochsNumber(1)
    nn5J_3DModel.setBatchSize(32)
    nn5J_3DModel.finalizeAdamModel(0.001)
    modelsList.append(nn5J_3DModel)
    nn5J_3DModel.getModel().summary()

modelPerformances = KFoldCrossValidation().performKFoldCrossValidation(modelsList, nn5J_3D["training-set-input"],
                                                                        nn5J_3D["training-set-output"], nn5J_3D["test-set-input"],
                                                                            nn5J_3D["test-set-output"])
modelsList.clear()

modelsPerformances.append(modelPerformances)
modelsPerformances.sort(key = lambda model : model.getFinalTestSetMSE(), reverse = False)

bestModel = modelsPerformances[0].getModel()
print("Model performances sorted by best to worst are the following:")
for modelPerformances in modelsPerformances:
    printModelDataAndPerformances(modelPerformances)

modelsPerformances.clear()