import numpy as np
from fastapi import FastAPI, File, UploadFile, Response
from PIL import Image
from io import BytesIO
from detection_service import FireDetector
import uvicorn
import cv2 as cv

app = FastAPI()
detector = FireDetector()
detector.load_image_detector()


@app.post("/detect")
async def detect_file(img: UploadFile = File(...)):
    valid_file = ['image/jpeg', 'image/jpg', 'image/png']
    if img.content_type not in valid_file:
        return {"error": "Invalid file content"}

    img_data = cv.imdecode(
        np.frombuffer(await img.read(), np.uint8), cv.IMREAD_COLOR)
    _, detections = detector.detect_from_array(img_data)
    return detections


@app.post("/process-image")
async def process_image(img: UploadFile = File(...)):
    valid_file = ['image/jpeg', 'image/jpg', 'image/png']
    if img.content_type not in valid_file:
        return {"error": "Invalid file content"}

    img_data = cv.imdecode(
        np.frombuffer(await img.read(), np.uint8), cv.IMREAD_COLOR)
    new_img = Image.fromarray(detector.detect_from_array(img_data)[0])
    output = BytesIO()
    new_img.save(output, 'png')
    return Response(output.getvalue(), media_type='image/png')


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
