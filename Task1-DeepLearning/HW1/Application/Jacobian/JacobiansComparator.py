import numpy as np
import tensorflow as tf
from NeuralNetwork.NeuralNetwork import NeuralNetwork

"""
This class provides the necessary methods to compute both the analytical Jacobian
and the calculated Jacobian.
"""
L1 = 0.1
L2 = 0.1

def computeAnalyticalJacobian(theta):
    theta1, theta2 = theta
    J = np.array([
        [-L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2), -L2 * np.sin(theta1 + theta2)],
        [L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2), L2 * np.cos(theta1 + theta2)]
    ])
    return J

def FK(model, theta):
    t = tf.reshape(theta, shape=(1, 2))
    out = model(t)
    out = tf.reshape(out, shape=(2,))
    return out

@tf.function
def FK_Jacobian(model, x):
    """ Necessary due to impossibility of correctly creating a clone 
        of a keras model without incurring in any kind of error or problem."""
    if isinstance(model, NeuralNetwork):
        model = model._getModel()
        
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = FK(model, x)

    return tape.jacobian(y, x)
