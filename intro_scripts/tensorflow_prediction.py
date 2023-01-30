from numpy.lib.npyio import load
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import os

def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(224,224))

    # convert img to array
    img = img_to_array(img)

    # reshape img into 3 channels
    img = img.reshape(1, 224, 224, 3)

    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]

    return img

# load the model
model = tf.keras.models.load_model(os.path.join("..", "final_model"))

img = load_image("../bronte.jpg")
prediction = model.predict(img)

if prediction[0][0] > 0.5: # 0 to 0.5 is cat and 0.5 to 1 is dog
    print("Pic of dog")
else:
    print("Pic of cat")
