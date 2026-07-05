import tensorflow as tf

model = tf.keras.models.load_model(
    "food_classifier_final.keras"
)

print("Model Loaded Successfully")
print(model.summary())