from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf

train_data_dir = "train"
validation_data_dir = "validation"
img_width, img_height = 100, 100
batch_size = 32
epochs = 10
kernel_size = (10, 10)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    interpolation="lanczos",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    interpolation="lanczos",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
)

model = Sequential()
model.add(
    Conv2D(
        32,
        kernel_size,
        activation="relu",
        input_shape=(img_width, img_height, 3),
    )
)
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, kernel_size, activation="relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, kernel_size, activation="relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
)

loss, accuracy = model.evaluate(
    validation_generator, steps=validation_generator.samples // batch_size
)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

model.save("./model/classification_model.keras")
