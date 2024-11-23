from NeuralNetwork.FKNNFactory import FKNNFactory

nn2J_2D = FKNNFactory().create2J_2D_NN()
nn2J_2D.addDenseLayer(64)
nn2J_2D.addDenseLayer(64)
nn2J_2D.addDenseLayer(64)
nn2J_2D.fitAdam(0.001)

nn3J_2D = FKNNFactory().create3J_2D_NN()
nn3J_2D.addDenseLayer(64)
nn3J_2D.addDenseLayer(64)
nn3J_2D.addDenseLayer(64)
nn3J_2D.fitAdam(0.001)

nn5J_3D = FKNNFactory().create5J_3D_NN()
nn5J_3D.addDenseLayer(64)
nn5J_3D.addDenseLayer(64)
nn5J_3D.addDenseLayer(64)
nn5J_3D.fitAdam(0.001)