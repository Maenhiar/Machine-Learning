from NeuralNetwork.FKNNFactory import FKNNFactory
from PerformaceEvaluation.KFoldCrossValidation import KFoldCrossValidation


modelsList = []
for i in range(3):
    nn2J_2D = FKNNFactory().create2J_2D_NN()
    nn2J_2D.addDenseLayer(64)
    nn2J_2D.addDenseLayer(64)
    nn2J_2D.addDenseLayer(64)
    nn2J_2D.setEarlyStopping(10)
    nn2J_2D.setEpochsNumber(100)
    nn2J_2D.setBatchSize(32)
    nn2J_2D.finalizeAdamModel(0.005)
    modelsList.append(nn2J_2D)

num_models = len(modelsList)
print(num_models)

for item in modelsList:
    item.getModel().summary()

KFoldCrossValidation().performKFoldCrossValidation(modelsList)

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