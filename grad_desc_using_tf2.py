"""
Gradient Descent using Tensorflow
"""

from numpy.lib.function_base import gradient
import tensorflow as tf2

weights = tf2.Variable([tf2.random.normal()])
'''
tf.random.normal(
    shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None, name=None
)

Outputs random values from a normal/gaussian distribution as a "tf2.Tensor"
object
'''

'''
tf.Variable(
    initial_value=None, trainable=None, validate_shape=True,
    caching_device=None, name=None, variable_def=None, dtype=None,
    import_scope=None, constraint=None,
    synchronization=tf.VariableSynchronization.AUTO,
    aggregation=tf.compat.v1.VariableAggregation.NONE, shape=None
)


A TensorFlow variable is the recommended way to represent shared,
persistent state your program manipulates
'''

while True:
    with tf2.GradientTape() as g:
        # tf2.GradientTape() records operations for automatic differentiation
        loss = compute_loss(weights)
        gradient = g.gradient(loss, weights)

    weights -= (lr * gradient)  # lr = learning rate

##############################################################################

model = tf2.keras.Sequential([...])

# Pick your favorite optimizer:
optimizer = tf2.keras.optimizer.SGD()

while True:  # loop forever
    # forward pass through the network
    prediction = model(x)

    with tf2.GradientTape() as tape
        # compute the loss
        loss = compute_loss(y, prediction)

    # update the weights using the gradient:
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

