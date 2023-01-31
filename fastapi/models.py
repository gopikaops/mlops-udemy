import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

import torch
from torchvision import transforms
from torchvision import models
from torch import nn    

import io
from PIL import Image
import base64
import numpy as np
import os

class Models():
    def __init__(self) -> None:
        self.tf_model = tf.keras.models.load_model(os.path.join("final_model"))
        checkpoint = torch.load(os.path.join("catvdog.pt"), map_location=torch.device('cpu'))
        self.pytorch_model = models.densenet121(pretrained=False)
        self.pytorch_model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, 256),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Linear(256, 2),
                                        nn.LogSoftmax(dim=1))
        self.pytorch_model.parameters = checkpoint['parameters']
        self.pytorch_model.load_state_dict(checkpoint['state_dict'])
        self.pytorch_model.eval()

    def predict_tensorflow(self, img_array: np.ndarray) -> dict:
        prediction = self.tf_model.predict(img_array)
        prediction_value = prediction[0][0]
        if prediction > 0.5: # 0 to 0.5 is cat and 0.5 to 1 is dog
            return {'class': 'dog', 'value': float(prediction_value)}
        else:
            return {'class': 'cat', 'value': float(prediction_value)}

    def load_image_tf(self, img_b64: str) -> np.ndarray:
        img = Image.open(io.BytesIO(base64.b64decode(img_b64)))

        img = img.convert('RGB')
        img = img.resize((224, 224), Image.NEAREST)

        # convert img to array
        img = img_to_array(img)

        # reshape img into 3 channels
        img = img.reshape(1, 224, 224, 3)

        # center pixel data
        img = img.astype('float32')
        img = img - [123.68, 116.779, 103.939]

        return img

    def predict_pytorch(self, img_tensor: torch.Tensor) -> dict:
        prediction = torch.exp(self.pytorch_model(img_tensor))
        topconf, topclass = prediction.topk(1, dim=1)

        if topclass.item() == 1:
            return {'class': 'dog', 'confidence': str(topconf.item())}
        else:
            return {'class': 'cat', 'confidence': str(topconf.item())}

    def load_image_pytorch(self, img_b64: str) -> torch.Tensor:
        test_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
        img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
        image_tensor = test_transforms(img)
        image_tensor = image_tensor[None, :, :, :]
        return image_tensor
