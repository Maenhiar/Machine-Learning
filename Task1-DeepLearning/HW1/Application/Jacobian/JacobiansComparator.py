import tensorflow as tf

# Parametri del braccio
L1 = 0.1
L2 = 0.1

def compute_jacobian(model, theta1, theta2):
    """
    Calcola la matrice Jacobiana numerica utilizzando TensorFlow.
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(theta1)  # Registriamo theta1
        tape.watch(theta2)  # Registriamo theta2
        
        # Calcoliamo l'output della rete (x, y) per gli angoli theta1 e theta2
        output = model(tf.stack([theta1, theta2], axis=0))

    # Calcoliamo il Jacobiano come la derivata dell'output rispetto a theta1 e theta2
    jacobian = tape.jacobian(output, [theta1, theta2])
    return jacobian

# Eseguiamo il calcolo per un set di angoli di esempio
theta1_test = tf.Variable(0.5)  # Angolo theta1
theta2_test = tf.Variable(0.5)  # Angolo theta2

# Calcoliamo il Jacobiano numerico
jacobian_numerical = compute_jacobian(model, theta1_test, theta2_test)

# Jacobiano analitico (calcolato manualmente)
def jacobian_analytical(theta1, theta2):
    """
    Calcola la matrice Jacobiana analitica per un braccio robotico a 2 DOF.
    """
    J11 = -L1 * tf.sin(theta1) - L2 * tf.sin(theta1 + theta2)
    J12 = -L2 * tf.sin(theta1 + theta2)
    J21 = L1 * tf.cos(theta1) + L2 * tf.cos(theta1 + theta2)
    J22 = L2 * tf.cos(theta1 + theta2)
    
    return tf.stack([[J11, J12], [J21, J22]])

# Jacobiano analitico
J_analytical = jacobian_analytical(0.5, 0.5)

# Mostriamo i risultati
print("Jacobiano Analitico:\n", J_analytical)
print("\nJacobiano Numerico (TensorFlow):\n", jacobian_numerical[0].numpy())  # Prendiamo il primo (e unico) valore del risultato
print("\nDifferenza tra Jacobiani:\n", np.abs(J_analytical - jacobian_numerical[0].numpy()))
