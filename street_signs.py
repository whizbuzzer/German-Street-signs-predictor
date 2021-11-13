'''
German Street signs classifier.
GTSRB dataset needed for this. Download from:
https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
'''

# To skip debugging statements:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from deep_learning_models import streetsigns_model
from my_utils import split_data, order_test_set, create_generators


if __name__ == "__main__":
    # # Change accordingly:
    # path_to_data = "C:\\Users\\anipr\\Downloads\\GTSRB\\Train"
    # path_to_save_train = "C:\\Users\\anipr\\Downloads\\GTSRB\\train_data\\train"
    # path_to_save_val = "C:\\Users\\anipr\\Downloads\\GTSRB\\train_data\\val"
    # split_data(path_to_data, path_to_save_train=path_to_save_train,
    # path_to_save_val=path_to_save_val)

    # path_to_images = "C:\\Users\\anipr\\Downloads\\GTSRB\\Test"
    # path_to_csv = "C:\\Users\\anipr\\Downloads\\GTSRB\\Test.csv"
    # order_test_set(path_to_images=path_to_images, path_to_csv=path_to_csv)

    path_to_train = "C:\\Users\\anipr\\Downloads\\GTSRB\\train_data\\train"
    path_to_val = "C:\\Users\\anipr\\Downloads\\GTSRB\\train_data\\val"
    path_to_test = "C:\\Users\\anipr\\Downloads\\GTSRB\\Test"
    BATCH_SIZE = 64
    EPOCHS = 15
    LR = 0.0001

    train_gen, val_gen, test_gen = create_generators(
        BATCH_SIZE,
        path_to_train,
        path_to_val,
        path_to_test
        )  # Larger batch size can cause you to run out of memory
    n_classes = train_gen.num_classes

    TRAIN = False
    TEST = True

    if TRAIN:
        # For saving the best model:
        model_save_path = ".\\Models"
        chkpt_saver = ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )  # max mode saves the model with a higher val_accuracy than the prev one

        early_stop = EarlyStopping(monitor="val_accuracy", patience=10)
        # Stops training if val_accuracy doesn't increase after 10 epochs

        model = streetsigns_model(n_classes)

        # Optimizers can also be passed as follows:
        from tensorflow.keras.optimizers import Adam
        optmzr = Adam(learning_rate=LR)
        # In this way, you can modify the parameters of the optimizer

        model.compile(
            optimizer=optmzr,
            loss='categorical_crossentropy',
            metrics=['accuracy']
            )

        model.fit(
            train_gen,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=[chkpt_saver, early_stop]  # Could give multiple callbacks
            )

    if TEST:  # Using a saved model
        from tensorflow.keras.models import load_model
        model = load_model('.\\Models')
        model.summary()

        # Evaluating models:
        print("Evaluating validation set")
        model.evaluate(val_gen)

        print("Evaluating test set")
        model.evaluate(test_gen)