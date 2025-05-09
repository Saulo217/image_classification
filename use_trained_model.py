import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("./model/classification_model.keras")

img_width, img_height = 100, 100


def classify_image(image_path):
    try:
        img = image.load_img(image_path, target_size=(img_width, img_height))

        img_array = image.img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0)

        img_array /= 255.0

        prediction = model.predict(img_array)

        prob_class_0 = prediction[0][0]  # Probability of class_0
        prob_class_1 = prediction[0][1]  # Probability of class_1

        result = f"Class_0: {'Yes' if prob_class_0 > 0.5 else 'No'}, Probability: {prob_class_0:.4f}\n"
        result += f"Class_1: {'Yes' if prob_class_1 > 0.5 else 'No'}, Probability: {prob_class_1:.4f}"

        print(result)
        pred = ""

        if prob_class_0 == 1 and prob_class_1 == 0:
            pred = "fruta"
        elif prob_class_1 == 1 and prob_class_0 == 0:
            pred = "utensilio"
        else:
            pred = "outros"

        return pred

    except FileNotFoundError:
        return f"Error: Image not found at {image_path}"
    except Exception as e:
        return f"An error occurred: {e}"


classify_image("./to_classify/goku.jpg")
