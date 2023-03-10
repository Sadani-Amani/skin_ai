import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

# Define the input shape for your model
IMG_SIZE = 128

# Define a function to preprocess the image
def preprocess_image(image):
    # Load the image using PIL
    img = Image.open(image)
    
    # Resize the image to the required shape
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert the image to a numpy array
    img = img_to_array(img)
    
    # Preprocess the image
    img = preprocess_input(img)
    
    # Add a dimension to represent the batch size
    img = np.expand_dims(img, axis=0)
    
    # Return the preprocessed image
    return img