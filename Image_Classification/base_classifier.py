import json
import os
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from PIL import Image, ImageOps
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

cached_weights_path = os.path.expanduser(
    "~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5"
)
cached_class_index_path = os.path.expanduser("~/.keras/models/imagenet_class_index.json")
model_weights = cached_weights_path if os.path.exists(cached_weights_path) else "imagenet"
model = MobileNetV2(weights=model_weights)
last_conv_layer_name = "Conv_1"
gradcam_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(last_conv_layer_name).output, model.output]
)

imagenet_class_index = None
if os.path.exists(cached_class_index_path):
    with open(cached_class_index_path, "r", encoding="utf-8") as class_index_file:
        imagenet_class_index = json.load(class_index_file)


def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)\
    return np.expand_dims(img_array, axis=0)


def make_gradcam_heatmap(img_array, pred_index):
    with tf.GradientTape() as tape:
        conv_output, predictions = gradcam_model([img_array], training=False)
        class_channel = predictions[:, pred_index]

    gradients = tape.gradient(class_channel, conv_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap).numpy()
    if max_value == 0:
        return np.zeros(heatmap.shape, dtype=np.float32)

    return (heatmap / max_value).numpy()


def save_gradcam_overlay(image_path, heatmap):
    original_image = Image.open(image_path).convert("RGB")
    heatmap_image = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_image = heatmap_image.resize(original_image.size, Image.BILINEAR)
    colored_heatmap = ImageOps.colorize(
        heatmap_image,
        black="black",
        mid="orange",
        white="red"
    ).convert("RGB")

    overlay_image = Image.blend(original_image, colored_heatmap, alpha=0.4)
    output_path = f"{os.path.splitext(image_path)[0]}_gradcam.jpg"
    overlay_image.save(output_path, format="JPEG")
    return output_path


def decode_top_predictions(predictions, top=3):
    if imagenet_class_index is None:
        return decode_predictions(predictions, top=top)[0]

    top_indices = predictions[0].argsort()[-top:][::-1]
    return [
        (
            imagenet_class_index[str(index)][0],
            imagenet_class_index[str(index)][1],
            float(predictions[0][index])
        )
        for index in top_indices
    ]

def classify_image(image_path):
    try:
        img_array = load_and_preprocess_image(image_path)
        predictions = model.predict(img_array, verbose=0)
        decoded_predictions = decode_top_predictions(predictions, top=3)
        predicted_class_index = int(np.argmax(predictions[0]))
        heatmap = make_gradcam_heatmap(img_array, predicted_class_index)
        gradcam_output_path = save_gradcam_overlay(image_path, heatmap)

        print("\nTop-3 Predictions for", image_path)
        for i, (_, label, score) in enumerate(decoded_predictions):
            print(f"  {i + 1}: {label} ({score:.2f})")
        print("Grad-CAM saved to:", gradcam_output_path)
    except Exception as e:
        print(f"Error processing '{image_path}': {e}")

if __name__ == "__main__":
    print("Image Classifier (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename: ").strip()
        if image_path.lower() == "exit":
            print("Goodbye!")
            break
        classify_image(image_path)
