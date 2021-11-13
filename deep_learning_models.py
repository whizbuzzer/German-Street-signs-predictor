'''
Deep learning model classes and methods:
'''

# To skip debugging statements:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf2
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, InputLayer, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D, Flatten 

class MyModel:
    def __init__(self, x_train, y_train, x_test, y_test, model_name: str):
        
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.model_name = model_name

        # Normalization causes data to move faster towards the global minimum
        # of your cost function:
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255

        # To accomodate datasets to the input layer:
        self.x_train = np.expand_dims(self.x_train, axis=-1)  # dimension at the end
        self.x_test = np.expand_dims(self.x_test, axis=-1)

        # One-hot encoding labels to use categorical cross entropy:
        self.y_train = tf2.keras.utils.to_categorical(self.y_train, 10)
        self.y_test = tf2.keras.utils.to_categorical(self.y_test, 10)

        '''
        Approach 1: Sequential:
        '''
        if self.model_name == "sequential":
            print("\n\n\nRunning Sequential model:\n")
            '''
            Defining Sequential model:

            Different layers may be used in different orders...
            Following sequence is one approach:

            28x28 single channel grayscale images to the input layer.
            One dimension is hidden and that extra '1' is because Conv2D() layer
            accepts a 4D tensor as input

            Conv2D() layer has 32 filters, each of size 3x3
            '''

            '''
            When using InputLayer with the Keras Sequential model,
            it can be skipped by moving the input_shape parameter to the
            first layer after the InputLayer.
            '''
            self.model = tf2.keras.Sequential(
                [
                    # Input(shape=(28, 28, 1)),
                    # InputLayer(input_shape=(28, 28, 1)),
                    
                    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                    Conv2D(64, (3, 3), activation='relu'),
                    MaxPool2D(),
                    BatchNormalization(),

                    Conv2D(128, (3, 3), activation='relu'),
                    MaxPool2D(),
                    BatchNormalization(),

                    GlobalAvgPool2D(),
                    Dense(64, activation='relu'),
                    Dense(10, activation='softmax')
                ]
            )
            '''
            GlobalAvgPool2D() computes the avg or values from above layer
            acc. to some axis
            '''

        '''
        Approach 2: Functional:
        '''
        if self.model_name == "functional":
            print("\n\n\nRunning Functional model:\n")
            self.model = self.functional_model()

        '''
        Approach 3: Inheriting from tensorflow.keras.model:
        '''
        if self.model_name == "custom":
            print("\n\n\nRunning Custom model:\n")
            self.model = CustomModel()

        '''
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        https://www.tensorflow.org/api_docs/python/tf/keras/losses
        https://www.tensorflow.org/api_docs/python/tf/keras/metrics
        '''
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics='accuracy'
            )

        # Model training:
        self.model.fit(
            self.x_train, self.y_train,
            batch_size=64,
            epochs=3,
            validation_split=0.2
            )
        # Dataset could be split into 3 subsets: train, validation and test

        # Evaluation on test set:
        self.model.evaluate(self.x_test, self.y_test, batch_size=64)
    

    def functional_model(self):
        '''
        In Functional approach, output of each layer is fed as input to the
        next layer as so:
        '''
        my_input = Input(shape=(28, 28, 1))
        x = Conv2D(32, (3, 3), activation='relu')(my_input)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)

        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)

        x = GlobalAvgPool2D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(10, activation='softmax')(x)

        model = tf2.keras.Model(inputs=my_input, outputs=x)

        return model
    

'''
Approach 3: Inheriting from tensorflow.keras.model:
'''
class CustomModel(tf2.keras.Model):
    def __init__(self):
        # "super()" borrows the constructor/initializer of the parent class:
        super().__init__()

        self.conv1 = Conv2D(32, (3, 3), activation='relu')
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.maxpool1 = MaxPool2D()
        self.batchnorm1 = BatchNormalization()

        self.conv3 = Conv2D(128, (3, 3), activation='relu')
        self.maxpool2 = MaxPool2D()
        self.batchnorm2 = BatchNormalization()

        self.globalavgpool1 = GlobalAvgPool2D()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    
    def call(self, my_input):
        x = self.conv1(my_input)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)
        x = self.globalavgpool1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        
        return x


def streetsigns_model(n_classes):
    my_input = Input(shape=(60, 60, 3))
    x = Conv2D(32, (3, 3), activation='relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)  # 5 * 5 * 128 = 3200 
    # x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(n_classes, activation='softmax')(x)

    return tf2.keras.Model(inputs=my_input, outputs=x)

if __name__ == "__main__":
    model = streetsigns_model(10)
    model.summary()  # Shows structure of the layers