import numpy as np
import tensorflow as tf
from Charts.LossPlotter import LossPlotter
from NeuralNetwork.FKNNFactory import FKNNFactory
from keras.layers import InputLayer
from PerformaceEvaluation.KFoldCrossValidation import KFoldCrossValidation
from Jacobian.JacobiansComparator import FK_Jacobian, computeAnalyticalJacobian
from NeuralNetwork.AverageModelProperties import AverageModelProperties

def computeJacobians(bestModel):
    theta = tf.constant([0.5, 1.0], dtype=tf.float32)
    computedJacobian = FK_Jacobian(bestModel, theta)
    numpyTheta = theta.numpy()
    jacobian_analytical = computeAnalyticalJacobian(numpyTheta)
    
    print("Jacobiano analitico:\n", jacobian_analytical)
    print("Jacobiano computato:\n", computedJacobian.numpy())
    print("Differenza tra i Jacobiani:", np.abs(computedJacobian.numpy() - jacobian_analytical))

def printModelDataAndPerformances(model):
    model.showModelSummary()

    print("Final MSE = ", model.getFinalMSE())
    print("Final test ste MSE = ", model.getFinalTestSetMSE())
    print(f'Training time: {model.getTrainingTime():.4f} secondi')
    
    print("\Model layers:")
    for layer in model.getModelLayers():
        if isinstance(layer, InputLayer):
            print(f"Layer: {layer.name}, Tipo: {layer.__class__.__name__}, Activation function: N/A")
        else:
            print(f"Layer: {layer.name}, Tipo: {layer.__class__.__name__}, Activation function: {getattr(layer, 'activation', None).__name__ if getattr(layer, 'activation', None) else 'N/A'}")

        if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
            reg_type = layer.kernel_regularizer.get_config()
            if 'l1' in reg_type and 'l2' not in reg_type:
                print(f"Layer {layer.name} has L1 regularization.")
            elif 'l2' in reg_type and 'l1' not in reg_type:
                print(f"Layer {layer.name} has L2 regularization.")
            elif 'l1' in reg_type and 'l2' in reg_type:
                print(f"Layer {layer.name} has L1 and L2 regularization.")

    print(f"Loss function: {model.getModelLoss()}")

    optimizer_config = model.getOptimizerConfiguration()
    print("Optimizer parameters:")
    for param, value in optimizer_config.items():
        print(f"  {param}: {value}")
    
    print("Early stopping: ", model.getEarlyStopping())

#2J-2D
modelsPerformances = []
modelsList = []
"""for i in range(3):
    nn2J_2D = FKNNFactory().create2J_2D_NN(True)
    nn2J_2DModel = nn2J_2D["modelWrapper"]
    nn2J_2DModel.addDenseLayer(64)
    nn2J_2DModel.addDenseLayer(64)
    nn2J_2DModel.addDenseLayer(64)
    nn2J_2DModel.setEpochsNumber(1)
    nn2J_2DModel.setBatchSize(64)
    nn2J_2DModel.setEarlyStopping(20)
    nn2J_2DModel.finalizeAdamModel(0.001)
    modelsList.append(nn2J_2DModel)

modelPerformances = KFoldCrossValidation().performKFoldCrossValidation(modelsList, nn2J_2D["training-set-input"],
                                                                        nn2J_2D["training-set-output"], nn2J_2D["test-set-input"],
                                                                            nn2J_2D["test-set-output"])

modelsList.clear()

modelProperties = AverageModelProperties(nn2J_2DModel, modelPerformances["avgMSE"], modelPerformances["avgTestSetMSE"], 
                                    modelPerformances["trainingTime"], modelPerformances["histories"])

modelsPerformances.append(modelProperties)

modelsPerformances.sort(key = lambda modelProperties : modelProperties.getFinalTestSetMSE(), reverse = False)

print("Model performances sorted by best to worst are the following:")
for modelPerformances in modelsPerformances:
    printModelDataAndPerformances(modelPerformances)

bestModel = modelsPerformances[0]

LossPlotter.plotLoss(bestModel.getmodelHistories()[0])

computeJacobians(bestModel.getFKNNModel())

modelsPerformances.clear()"""

#3J-2D
for i in range(3):
    nn3J_2D = FKNNFactory().create3J_2D_NN(True)
    nn3J_2DModel = nn3J_2D["modelWrapper"]
    nn3J_2DModel.addDenseLayer(64)
    nn3J_2DModel.addDenseLayer(64)
    nn3J_2DModel.addDenseLayer(64)
    nn3J_2DModel.setEpochsNumber(100)
    nn3J_2DModel.setBatchSize(64)
    nn3J_2DModel.finalizeAdamModel(0.001)
    nn3J_2DModel.setEarlyStopping(50)
    modelsList.append(nn3J_2DModel)

modelPerformances = KFoldCrossValidation().performKFoldCrossValidation(modelsList, nn3J_2D["training-set-input"],
                                                                        nn3J_2D["training-set-output"], nn3J_2D["test-set-input"],
                                                                            nn3J_2D["test-set-output"])

modelsList.clear()

modelProperties = AverageModelProperties(nn3J_2DModel, modelPerformances["avgMSE"], modelPerformances["avgTestSetMSE"], 
                                    modelPerformances["trainingTime"], modelPerformances["histories"])

modelsPerformances.append(modelProperties)

modelsPerformances.sort(key = lambda modelProperties : modelProperties.getFinalTestSetMSE(), reverse = False)

print("Model performances sorted by best to worst are the following:")
for modelPerformances in modelsPerformances:
    printModelDataAndPerformances(modelPerformances)

LossPlotter.plotLoss(modelsPerformances[0].getmodelHistories()[0])

modelsPerformances.clear()

#5G-3D
for i in range(3):
    nn5J_3D = FKNNFactory().create5J_3D_NN(True)
    nn5J_3DModel = nn5J_3D["modelWrapper"]
    nn5J_3DModel.addDenseLayer(64)
    nn5J_3DModel.addDenseLayer(64)
    nn5J_3DModel.addDenseLayer(64)
    nn5J_3DModel.setEpochsNumber(100)
    nn5J_3DModel.setBatchSize(32)
    nn5J_3DModel.setEarlyStopping(50)
    nn5J_3DModel.finalizeAdamModel(0.001)
    modelsList.append(nn5J_3DModel)

modelPerformances = KFoldCrossValidation().performKFoldCrossValidation(modelsList, nn5J_3D["training-set-input"],
                                                                        nn5J_3D["training-set-output"], nn5J_3D["test-set-input"],
                                                                            nn5J_3D["test-set-output"])
                                                                        
modelsList.clear()

modelProperties = AverageModelProperties(nn5J_3DModel, modelPerformances["avgMSE"], modelPerformances["avgTestSetMSE"], 
                                    modelPerformances["trainingTime"], modelPerformances["histories"])

modelsPerformances.append(modelProperties)

modelsPerformances.sort(key = lambda modelProperties : modelProperties.getFinalTestSetMSE(), reverse = False)

print("Model performances sorted by best to worst are the following:")
for modelPerformances in modelsPerformances:
    printModelDataAndPerformances(modelPerformances)
    
LossPlotter.plotLoss(modelsPerformances[0].getmodelHistories()[0])

modelsPerformances.clear()