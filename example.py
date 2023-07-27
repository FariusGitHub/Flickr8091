import io
import urllib
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import efficientnet
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import joblib
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

IMAGE_SIZE = (200,200)
# url = "https://raw.githubusercontent.com/FariusGitHub/DataScience/master/slack1.png"
url = "https://raw.githubusercontent.com/FariusGitHub/Flickr8k/master/58357057_dea882479e.jpg"
# url = 'https://github.com/FariusGitHub/Flickr8K/blob/main/3637013_c675de7705.jpeg'
def decode_and_resize(url):
    img = tf.constant(io.BytesIO(urllib.request.urlopen(url).read()).read())
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.expand_dims(img, 0)

def get_cnn_model(dummy):
    base_model = efficientnet.EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet",
    )
    # We freeze our feature extractor
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model

class ImageCaptioningModel(keras.Model):
    def __init__(
        self, cnn_model
    ):
        super().__init__()
        self.cnn_model = cnn_model
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")

def testing(img_path):
  return ImageCaptioningModel(get_cnn_model('')).cnn_model(decode_and_resize(img_path))


testing(url)
