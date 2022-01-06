"""
Class for Dense/Fully connected layer
Reference: https://www.youtube.com/watch?v=5tvmMX8r_OM&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=1
"""

import tensorflow as tf2


class MyDenseLayer(tf2.keras.layers.Layer):  # Inherited class
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__()  # inheriting from parent class

        # Initialize weights and bias:
        self.W = self.add_weight([input_dim, output_dim])
        self.b = self.add_weight([1, output_dim])

    
    def call(self, inputs):
        # Forward propagate the inputs
        z = tf2.matmul(inputs, self.W) + self.b

        # Feed through a non-linear activation:
        output = tf2.math.sigmoid(z)
        return output


"""
Tensorflow already has a dense layer implementation as:
layer = tf2.keras.layers.Dense(units=2)
"""