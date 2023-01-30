from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from logging.config import dictConfig
from logger_config import log_config
from models import Models

app = FastAPI()
models = Models()
dictConfig(log_config)
logger = logging.getLogger("mlops")

class ImagePayload(BaseModel):
    img_b64: str


@app.get('/health')
def health():
    logger.info("Health request received.")
    return "Service is online."

@app.post('/classify/tensorflow')
def classify_tensorflow(request: ImagePayload):
    try:
        logger.info("Tensorflow request received.")
        img_array = models.load_image_tf(request.img_b64)
        result = models.predict_tensorflow(img_array)
        return result
    except Exception as e:
        message = "Server error while processing image!"
        logger.error(f"{message}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=message)

@app.post('/classify/pytorch')
def classify_pytorch(request: ImagePayload):
    try:
        logger.info("PyTorch request received.")
        img_tensor = models.load_image_pytorch(request.img_b64)
        result = models.predict_pytorch(img_tensor)
        return result
    except Exception as e:
        message = "Server error while processing image!"
        logger.error(f"{message}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=message)

