'''
Image classifier for German signs using Tensorflow 2
'''

# To skip debugging statements:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.keras.datasets.mnist import load_data
# from tensorflow.python.keras import activations

from my_utils import display_some_examples
from deep_learning_models import MyModel


if __name__ == "__main__":
    
    (x_train, y_train), (x_test, y_test) = load_data()
    '''
    a path where to cache the dataset locally (relative to ~/.keras/datasets)
    could be provided to above command
    '''

    # print("x_train.shape={}, y_train.shape={}".format(x_train.shape, y_train.shape))
    # print("x_test.shape={}, y_test.shape={}".format(x_test.shape, y_test.shape))
    '''
    x_train.shape=(60000, 28, 28), y_train.shape=(60000,)
    -> 28x28 pixel images with 60000 labels/classes

    x_test.shape=(10000, 28, 28), y_test.shape=(10000,)
    -> 28x28 pixel images with 10000 labels/classes
    '''
    if False:
        display_some_examples(x_train, y_train)

    # print("x_train.shape={}, y_train.shape={}".format(x_train.shape, y_train.shape))
    # print("x_test.shape={}, y_test.shape={}".format(x_test.shape, y_test.shape))
    
    MyModel(x_train, y_train, x_test, y_test, "sequential")
    MyModel(x_train, y_train, x_test, y_test, "functional")
    MyModel(x_train, y_train, x_test, y_test, "custom")
    

    