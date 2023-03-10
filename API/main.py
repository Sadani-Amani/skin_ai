from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
from tensorflow.keras.models import load_model

app = FastAPI()

# Define the path to your trained model
# MODEL_PATH = './Resnet'
MODEL_PATH = os.path.dirname(__file__)+'/Resnet.h5'
# Load the trained model
model = load_model(MODEL_PATH)
# Define the input shape for your model
IMG_SIZE = 128

# Define a function to preprocess the image
def preprocess_image(image):
    # Load the image using PIL
    # img = Image.open(image)
    
    # Resize the image to the required shape
    nparr = np.fromstring(image, np.uint8)
    img = nparr.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert the image to a numpy array
    img = img_to_array(img)
    
    # Preprocess the image
    img = preprocess_input(img)
    
    # Add a dimension to represent the batch size
    img = np.expand_dims(img, axis=0)
    
    # Return the preprocessed image
    return img


@app.get("/")
def root ():
    return "API works"


@app.post("/SkinAI")
async def pred (img : UploadFile=File(...)):
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Read the image file
    # image_contents = await img.read()
    
    # Preprocess the image
    # img = preprocess_image(image_contents)
    processed_image = cv2_img.resize(128,128,3)
    processed_image = preprocess_input(cv2_img)
    processed_image = np.expand_dims(processed_image, axis=0)
    # Make a prediction
    preds = model.predict(processed_image)
    
    # Get the predicted class
    predicted_class = np.argmax(preds)
    
    # Define the class labels
    classes = ['Acne and Rosacea Photos', 'Atopic Dermatitis Photos',
                         'Exanthems and Drug Eruptions', 'Hair Loss Photos Alopecia and other Hair Diseases',
                         'Herpes HPV and other STDs Photos', 'Lupus and other Connective Tissue diseases',
                         'Melanoma Skin Cancer Nevi and Moles', 'Poison Ivy Photos and other Contact Dermatitis',
                         'Scabies Lyme Disease and other Infestations and Bites', 'Vascular Tumors',
                         'Vasculitis Photos', 'Urticaria Hives']
    
    # Return the predicted class as a JSON response
    return {'class': classes[predicted_class]}

    
# Run the app
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)




