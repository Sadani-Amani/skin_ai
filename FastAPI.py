from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import cv2
import os
import json
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow import convert_to_tensor
from PIL import Image
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define the path to your trained model
# MODEL_PATH = './Resnet'
MODEL_PATH = os.path.dirname(__file__)+'/final_model.h5'
# Load the trained model
model = load_model(MODEL_PATH)


# Define the input shape for your model
IMG_SIZE = 128

# Define a function to preprocess the image
def preprocess_image(image):
    pass

@app.get("/")
def root ():
    return "API works"


@app.post("/SkinAI")
async def pred (img : UploadFile=File(...)):
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    cv2_img = cv2.resize(cv2_img,(IMG_SIZE,IMG_SIZE))
    arr = np.asarray(cv2_img)
    img = np.expand_dims(arr, axis=0)
    img = img/255.0
    # return img.shape
    # # Read the image file
    # # image_contents = await img.read()
    # img = img_to_array(cv2_img)
    # img = img.resize(img, (IMG_SIZE,IMG_SIZE,3))
    # img = np.expand_dims(img, axis=0)
    # img = img/255.0
    # # Make a prediction
    # # Get the predicted class
    # classes = [
    #     'Acne and Rosacea Photos',
    #     'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    #     'Atopic Dermatitis Photos',
    #     'Bullous Disease Photos',
    #     'Cellulitis Impetigo and other Bacterial Infections',
    #     'Eczema Photos',
    #     'Exanthems and Drug Eruptions',
    #     'Hair Loss Photos Alopecia and other Hair Diseases',
    #     'Herpes HPV and other STDs Photos',
    #     'Light Diseases and Disorders of Pigmentation',
    #     'Lupus and other Connective Tissue diseases',
    #     'Melanoma Skin Cancer Nevi and Moles',
    #     'Nail Fungus and other Nail Disease',
    #     'Poison Ivy Photos and other Contact Dermatitis',
    #     'Psoriasis pictures Lichen Planus and related diseases',
    #     'Scabies Lyme Disease and other Infestations and Bites',
    #     'Seborrheic Keratoses and other Benign Tumors',
    #     'Systemic Disease',
    #     'Tinea Ringworm Candidiasis and other Fungal Infections',
    #     'Urticaria Hives',
    #     'Vascular Tumors',
    #     'Vasculitis Photos',
    #     'Warts Molluscum and other Viral Infections'
    # ]
    preds = model.predict(img)
    return np.argmax(preds).tolist()

    # # Return the predicted class as a JSON response
    # return classes[np.argmax(preds).tolist()]

# Run the app
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
