'''
Utility functions:
'''
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import shutil
import csv


def display_some_examples(examples, labels):
    plt.figure(figsize=(10, 10))

    # plotting 25 images (arbitrary number):
    for i in range(25):
        idx = np.random.randint(0, examples.shape[0] - 1)
        img, label = examples[idx], labels[idx]
        plt.subplot(5, 5, i+1)  # Plotting i+1 images in a 5x5 grid
        plt.title(str(label))
        plt.imshow(img, cmap='gray')
    
    plt.show()


# Function to split the data for training:
def split_data(path_to_data, path_to_save_train, path_to_save_val,
    split_size=0.1):  # 90/10 split between train/validation
    folders = os.listdir(path_to_data)

    for folder in folders:
        full_path = os.path.join(path_to_data, folder)
        images_paths = glob.glob(os.path.join(full_path, '*.png'))
        
        # Splitting the dataset between training and validation
        x_train, x_val = train_test_split(images_paths, test_size=split_size)

        # For train set:
        for x in x_train:
            # basename = os.path.basename(x)
            path_to_folder = os.path.join(path_to_save_train, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)

        # For validation set:
        for x in x_val:
            path_to_folder = os.path.join(path_to_save_val, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)


def order_test_set(path_to_images, path_to_csv):
    # testset = {}
    try:
        with open(path_to_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            for i, row in enumerate(reader):
                # Avoiding first row as it just contains the tags
                if i == 0:
                    continue
                
                # Removing redundant "Test/" part of the image name:
                img_name = row[-1].replace('Test/', '')

                # Second-last element of a row is class id:
                label = row[-2]

                path_to_folder = os.path.join(path_to_images, label)

                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)

                img_full_path = os.path.join(path_to_images, img_name)
                shutil.move(img_full_path, path_to_folder)

    except:
        print('[INFO] : Error reading CSV file')


def create_generators(batch_size, train_data_path, test_data_path, val_data_path):
    '''
    Optimization can be done with the model as well as with the data.
    On data, it could be done with preprocessors:
    '''
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_preprocessor = ImageDataGenerator(
        rescale = 1 / 255.,
        rotation_range=10,  # Rotating images by (-10, 10) degrees
        width_shift_range=0.1 # Shifting images by (-10%, 10%) horizontally
    )
    
    '''
    Don't use preprocessor with modifications like shifting/rotating
    for validation/test sets as such changes are synthetic and don't exist on
    real-world data
    '''
    test_preprocessor = ImageDataGenerator(
        rescale = 1 / 255.
    )


    train_generator = train_preprocessor.flow_from_directory(
        train_data_path,
        class_mode="categorical",
        target_size=(60, 60),
        color_mode="rgb",
        shuffle=True,  # For the randomness. Ensures model's robustness
        batch_size=batch_size
    )

    val_generator = test_preprocessor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(60, 60),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )

    test_generator = test_preprocessor.flow_from_directory(
        test_data_path,
        class_mode="categorical",
        target_size=(60, 60),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )

    return train_generator, val_generator, test_generator