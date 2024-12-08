import numpy as np
import tensorflow as tf
from Charts.LossPlotter import LossPlotter
from NeuralNetwork.FKNNFactory import FKNNFactory
from keras.layers import InputLayer
from PerformaceEvaluation.KFoldCrossValidation import KFoldCrossValidation
from Jacobian.JacobiansComparator import FK_Jacobian, computeAnalyticalJacobian, computeMatrixesDifference, computeNorm
from NeuralNetwork.AverageModelProperties import AverageModelProperties

def printJacobians(bestModel):
    theta = tf.constant([0.056, 0.021], dtype=tf.float32)
    computedJacobian = FK_Jacobian(bestModel, theta)
    numpyTheta = theta.numpy()
    jacobian_analytical = computeAnalyticalJacobian(numpyTheta)
    
    print("Analytical Jacobian:\n", jacobian_analytical)
    print("Model computed Jacobian:\n", computedJacobian.numpy())
    print("Difference:\n", computeMatrixesDifference(computedJacobian.numpy(), jacobian_analytical))
    print("Norm: ", computeNorm(computedJacobian.numpy(), jacobian_analytical))

def printModelDataAndPerformances(model):
    model.showModelSummary()

    print("Final average MSE = ", model.getFinalMSE())
    print("Final average test set MSE = ", model.getFinalTestSetMSE())
    print(f'Training time: {model.getTrainingTime():.4f} seconds')
    
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

modelsPerformances = []
modelsList = []
retrieveModelString = "modelWrapper"
retrieveTrainingSetInputString = "training-set-input"
retrieveTrainingSetOutputString = "training-set-output"
retrieveTestSetInputString = "test-set-input"
retrieveTestSeOutputString = "test-set-output"
retrieveAvgMSEString = "avgMSE"
retrieveAvgTestSetMSEString = "avgTestSetMSE"
retrieveTrainingTimeString = "trainingTime"
retrieveHistoryString = "histories"

#2J-2D
for i in range(3):
    nn2J_2D = FKNNFactory().create2J_2D_NN(True)
    nn2J_2DModel = nn2J_2D[retrieveModelString]
    nn2J_2DModel.addDenseLayer(64)
    nn2J_2DModel.addDenseLayer(64)
    nn2J_2DModel.addDenseLayer(64)
    nn2J_2DModel.addDenseLayer(64)
    nn2J_2DModel.addDenseLayer(64)
    nn2J_2DModel.setEpochsNumber(100)
    nn2J_2DModel.setBatchSize(64)
    nn2J_2DModel.setEarlyStopping(20)
    nn2J_2DModel.finalizeAdamModel(0.001)
    modelsList.append(nn2J_2DModel)

modelPerformances = KFoldCrossValidation().performKFoldCrossValidation(modelsList, nn2J_2D[retrieveTrainingSetInputString],
                                                                        nn2J_2D[retrieveTrainingSetOutputString], 
                                                                            nn2J_2D[retrieveTestSetInputString],
                                                                                nn2J_2D[retrieveTestSeOutputString])

modelsList.clear()

modelProperties = AverageModelProperties(nn2J_2DModel, modelPerformances[retrieveAvgMSEString], 
                                            modelPerformances[retrieveAvgTestSetMSEString], 
                                                modelPerformances[retrieveTrainingTimeString],
                                                    modelPerformances[retrieveHistoryString])

modelsPerformances.append(modelProperties)

modelsPerformances.sort(key = lambda modelProperties : modelProperties.getFinalTestSetMSE(), reverse = False)

print("Model performances sorted by best to worst are the following:")
for modelPerformances in modelsPerformances:
    printModelDataAndPerformances(modelPerformances)

bestModel = modelsPerformances[0]

LossPlotter.plotLoss(bestModel.getmodelHistories()[0])

printJacobians(bestModel.getFKNNModel())

modelsPerformances.clear()

#3J-2D
for i in range(3):
    nn3J_2D = FKNNFactory().create3J_2D_NN(True)
    nn3J_2DModel = nn3J_2D[retrieveModelString]
    nn3J_2DModel.addDenseLayer(64)
    nn3J_2DModel.addDenseLayer(64)
    nn3J_2DModel.addDenseLayer(64)
    nn3J_2DModel.addDenseLayer(64)
    nn3J_2DModel.addDenseLayer(64)
    nn3J_2DModel.setEpochsNumber(100)
    nn3J_2DModel.setBatchSize(64)
    nn3J_2DModel.finalizeAdamModel(0.001)
    nn3J_2DModel.setEarlyStopping(50)
    modelsList.append(nn3J_2DModel)

modelPerformances = KFoldCrossValidation().performKFoldCrossValidation(modelsList, nn3J_2D[retrieveTrainingSetInputString],
                                                                        nn3J_2D[retrieveTrainingSetOutputString], 
                                                                            nn3J_2D[retrieveTestSetInputString],
                                                                                nn3J_2D[retrieveTestSeOutputString])

modelsList.clear()

modelProperties = AverageModelProperties(nn3J_2DModel, modelPerformances[retrieveAvgMSEString], 
                                         modelPerformances[retrieveAvgTestSetMSEString], 
                                            modelPerformances[retrieveTrainingTimeString],
                                                modelPerformances[retrieveHistoryString])

modelsPerformances.append(modelProperties)

print("Model performances sorted by best to worst are the following:")
for modelPerformances in modelsPerformances:
    printModelDataAndPerformances(modelPerformances)
    
LossPlotter.plotLoss(modelsPerformances[0].getmodelHistories()[0])

modelsPerformances.clear()

#5G-3D
for i in range(3):
    nn5J_3D = FKNNFactory().create5J_3D_NN(True)
    nn5J_3DModel = nn5J_3D[retrieveModelString]
    nn5J_3DModel.addDenseLayer(64)
    nn5J_3DModel.addDenseLayer(64)
    nn5J_3DModel.addDenseLayer(64)
    nn5J_3DModel.setEpochsNumber(100)
    nn5J_3DModel.setBatchSize(64)
    nn5J_3DModel.setEarlyStopping(50)
    nn5J_3DModel.finalizeAdamModel(0.001)
    modelsList.append(nn5J_3DModel)

modelPerformances = KFoldCrossValidation().performKFoldCrossValidation(modelsList, nn5J_3D[retrieveTrainingSetInputString],
                                                                        nn5J_3D[retrieveTrainingSetOutputString], 
                                                                            nn5J_3D[retrieveTestSetInputString],
                                                                                nn5J_3D[retrieveTestSeOutputString])

modelsList.clear()

modelProperties = AverageModelProperties(nn5J_3DModel, modelPerformances[retrieveAvgMSEString], 
                                         modelPerformances[retrieveAvgTestSetMSEString], 
                                            modelPerformances[retrieveTrainingTimeString],
                                                modelPerformances[retrieveHistoryString])

modelsPerformances.append(modelProperties)

modelsPerformances.sort(key = lambda modelProperties : modelProperties.getFinalTestSetMSE(), reverse = False)

print("Model performances sorted by best to worst are the following:")
for modelPerformances in modelsPerformances:
    printModelDataAndPerformances(modelPerformances)
    
LossPlotter.plotLoss(modelsPerformances[0].getmodelHistories()[0])

modelsPerformances.clear()
