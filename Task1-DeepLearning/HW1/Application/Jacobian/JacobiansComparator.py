import numpy as np
import tensorflow as tf

# Parametri del manipolatore
L1 = 0.1
L2 = 0.1

# Jacobiano analitico
def Jacobian_analytical(theta):
    theta1, theta2 = theta
    J = np.array([
        [-L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2), -L2 * np.sin(theta1 + theta2)],
        [L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2), L2 * np.cos(theta1 + theta2)]
    ])
    return J

# Funzione di Forward Kinematics
def FK(model, theta):
    # Ridimensiona theta per batch di dimensione 1
    t = tf.reshape(theta, shape=(1, 2))  # theta ha dimensione (2,) per un problema 2D
    out = model(t)  # Passa l'input al modello (reti neurale)
    # Reshape dell'output per ottenere un vettore 1D con le coordinate finali
    out = tf.reshape(out, shape=(2,))
    return out

# Calcolo del Jacobiano usando TensorFlow
@tf.function
def FK_Jacobian(model, x):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)  # Teniamo traccia delle derivate rispetto a x
        y = FK(model, x)  # Calcola la posizione finale dal modello
    # Restituisci il Jacobiano, che è la derivata della posizione rispetto a x
    return tape.jacobian(y, x)
