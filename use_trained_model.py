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

        if prediction[0] > 0.5:
            predicted_class = "class_1"
            probability = prediction[0][0]
        else:
            predicted_class = "class_0"
            probability = 1 - prediction[0][0]

        return f"Predicted class: {predicted_class}, Probability: {probability:.4f}"

    except FileNotFoundError:
        return f"Error: Image not found at {image_path}"
    except Exception as e:
        return f"An error occurred: {e}"


print(classify_image("./to_classify/bg.jpg"))
print(classify_image("./to_classify/tom.jpg"))
print(classify_image("./to_classify/pa.png"))
print(classify_image("./to_classify/pedra.jpg"))
print(classify_image("./to_classify/pedra_es.jpg"))
print(classify_image("./to_classify/lim.jpg"))
