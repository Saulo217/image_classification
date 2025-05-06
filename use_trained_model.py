import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("./model/classification_model.h5")

# Define the target image size (must match the training image size)
img_width, img_height = 100, 100


def classify_image(image_path):
    """
    Loads an image, preprocesses it, and uses the trained model to predict its class.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The predicted class label ('class_0' or 'class_1') and the probability.
    """
    try:
        # Load the image
        img = image.load_img(image_path, target_size=(img_width, img_height))

        # Convert the image to a numpy array
        img_array = image.img_to_array(img)

        # Expand the dimensions to create a batch of size 1
        img_array = np.expand_dims(img_array, axis=0)

        # Normalize the image (as done during training)
        img_array /= 255.0

        # Make the prediction
        prediction = model.predict(img_array)

        # Interpret the prediction (assuming binary classification)
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


# Example usage:
print(classify_image("./to_classify/fork.png"))
print(classify_image("./to_classify/spoon.jpg"))
print(classify_image("./to_classify/fork1.jpg"))
print(classify_image("./to_classify/knife.jpg"))
print(classify_image("./to_classify/trident_001.jpg"))
