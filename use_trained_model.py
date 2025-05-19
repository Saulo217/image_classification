import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from io import BytesIO
import os
import numpy as np

img_width, img_height = 100, 100


def interpret_prediction(prediction, class_labels=None, threshold=0.75):
    """
    Generalizes interpretation of prediction vector with threshold handling.

    :param prediction: Output from model.predict(), shape (1, num_classes)
    :param class_labels: Optional list of class names (e.g., ["fruta", "utensilio", "outros"])
    :param threshold: Probability threshold to consider "Yes"
    :return: result dict with class-wise predictions and final label
    """
    probs = prediction[0]  # Assuming prediction shape is (1, num_classes)
    num_classes = len(probs)

    # Default labels if not provided
    if class_labels is None:
        class_labels = [f"Class_{i}" for i in range(num_classes)]

    result = {}
    max_class_index = int(np.argmax(probs))  # Get the index of the max probability
    final_pred_label = class_labels[max_class_index]

    # Find all classes with probabilities above the threshold
    above_threshold = [
        probs for i, prob in enumerate(probs) if round(prob, 2) > threshold
    ]

    print(above_threshold)
    # If no class exceeds the threshold, we can choose how to handle it.
    if not above_threshold:
        final_pred_label = "Outros"  # Return "Outros" if none exceed the threshold

    for i in range(num_classes):
        result[class_labels[i]] = {
            "Yes/No": "Yes" if probs[i] > threshold else "No",
            "Probability": float(probs[i]).__round__(2),
        }

    # Return both the result (class-wise) and the final label (with threshold consideration)
    return result, final_pred_label


def classify_image(file):
    model = tf.keras.models.load_model("./model/classification_model.keras")
    try:
        img = Image.open(BytesIO(file.read()))
        img.load()
        img = img.convert("RGB")
        img = img.resize((img_width, img_height))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)

        """
        class_labels = sorted(
            [
                name
                for name in os.listdir("./validation")
                if os.path.isdir(os.path.join("./validation", name))
            ]
        )"""
        class_labels = ["emotions", "fruits", "person", "toys", "utensilio"]

        result, pred = interpret_prediction(prediction, class_labels)

        return result, pred

    except Exception as e:
        return {"error": str(e)}, None
