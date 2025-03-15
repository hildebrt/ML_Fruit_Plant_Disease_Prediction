import pickle
import json
from PIL import Image
import numpy as np


class_indices = json.load(open('class_indices.json'))

model = pickle.load(open('model.pkl', 'rb'))


def load_and_preprocess_image(image_path, target_size=(224,224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.

    return img_array


def predict_image_class(image_path):
    preprocessed_img = load_and_preprocess_image(image_path=image_path)
    prediction = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]

    return predicted_class_name
