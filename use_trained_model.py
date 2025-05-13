import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from io import BytesIO

img_width, img_height = 100, 100

def classify_image(file):
    model = tf.keras.models.load_model("./model/classification_model.keras")
    try:
        img = Image.open(BytesIO(file.read()))
        img.load()
        img = img.resize((img_width, img_height))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)

        prob_class_0 = prediction[0][0]
        prob_class_1 = prediction[0][1]

        result = {
            "Class_0": "Yes" if prob_class_0 > 0.5 else "No",
            "Probability_Class_0": float(prob_class_0),
            "Class_1": "Yes" if prob_class_1 > 0.5 else "No",
            "Probability_Class_1": float(prob_class_1)
        }

        pred = ""
        if result["Class_0"] == "Yes" and result["Class_1"] == "No":
            pred = "fruta"
        elif result["Class_1"] == "Yes" and result["Class_0"] == "No":
            pred = "utensilio"
        else:
            pred = "outros"

        return result, pred

    except Exception as e:
        return {"error": str(e)}, None
