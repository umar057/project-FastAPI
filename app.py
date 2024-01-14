from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from joblib import load 
import numpy as np
from PIL import Image
import requests
from io import BytesIO

import os
import tempfile


app = FastAPI()

#1. use this if your model file is at some live hosting file manager
# URL of the model file
#model_url = 'https://epsoldevops.com/ML/model.h5'

# Download the model file
#response = requests.get(model_url)

# # Check if the request was successful (status code 200)
# if response.status_code == 200:
#     # Save the content to a temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
#         temp_file.write(response.content)
#         temp_file_path = temp_file.name

#     try:
#         # Load the model from the temporary file
#         model = load_model(temp_file_path)
#         print("Model loaded successfully.")
#     finally:
#         # Clean up: Remove the temporary file
#         temp_file.close()
#         # Uncomment the line below to delete the temporary file after loading the model
#         os.remove(temp_file_path)
# else:
#     print(f"Failed to download the model. Status code: {response.status_code}")



#2. use this if you have uploaded the file on google drive, and any
#any one with the link can view the file

# Replace 'FILE_ID' with the actual ID of the file in Google Drive
file_id = '13eUosz88gTTTh4W3kum3qTIqLo2w-FXm'

# Google Drive shared link format
drive_link = f'https://drive.google.com/export=download&id={file_id}'

# Download the model file
response = requests.get(drive_link)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Load the model from the content of the response
    model_content = BytesIO(response.content)
    model = load(model_content)
    print("Model loaded successfully.")
else:
    print(f"Failed to download the model. Status code: {response.status_code}")


#3. if you have file directly in the same folder as your .py file, use this
# Load the trained model
#model = load_model('model.h5')

# Define the request model using Pydantic
class PredictionRequest(BaseModel):
    image_path: str

# Define the image preprocessing function
def preprocess_image(image_path):
    try:
        if image_path.startswith('http'):
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(image_path)

        img = img.resize((224, 224))  # Adjust the size as needed
        img = np.array(img) / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading image: {str(e)}")

@app.post('/predict')
async def predict(request: PredictionRequest):
    try:
        # Preprocess the image
        processed_image = preprocess_image(request.image_path)

        # Make a prediction
        prediction = model.predict(processed_image)

        # Assuming binary classification, convert the prediction to a human-readable label
        label = "Glaucoma" if prediction > 0.5 else "Not Glaucoma"

        response = {'prediction': label}
        return JSONResponse(content=jsonable_encoder(response), status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
def index():
    return {"details":"Hello!!"}
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
