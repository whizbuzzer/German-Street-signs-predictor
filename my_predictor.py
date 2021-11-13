'''
Single image predictor
'''
# To skip debugging statements:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from my_utils import predict_with_model

if __name__ == "__main__":
    # Path of a random image:
    # img_path = "C:\\Users\\anipr\\Downloads\\GTSRB\\Test\\2\\00409.png"
    img_path = "C:\\Users\\anipr\\Downloads\\GTSRB\\Test\\0\\03420.png"

    # img_class = 2
    img_class = 0

    # Loading a saved model:
    from tensorflow.keras.models import load_model
    model = load_model('.\\Models')
    model.summary()

    predict = predict_with_model(model, img_path)
    print(f"prediction = {predict}")
    print("Prediction matched = ", predict == img_class)