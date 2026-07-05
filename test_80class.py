import tensorflow as tf

model = tf.keras.models.load_model(
    "food_classifier_80class.keras"
)

print("80 Class Model Loaded Successfully")