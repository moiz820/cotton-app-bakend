
from fastapi import  FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import  Image
import tensorflow as tf
app= FastAPI()

MODEL = tf.keras.models.load_model("models/test")
CLASS_NAMES=["bacterial_blight","curl_virus","fussarium_wilt","healthy"]
@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    Predictions=MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(Predictions[0])]
    confidence = np.max(Predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
