from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import requests
from io import BytesIO

app = FastAPI()

# Load the trained model
model = load_model('model.h5')

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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
