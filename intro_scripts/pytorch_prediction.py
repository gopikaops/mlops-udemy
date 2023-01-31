import torch
from PIL import Image
from torchvision import transforms
from torchvision import models
from torch import nn

def image_transformation(imagepath):
    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
    image = Image.open(imagepath)
    imagetensor = test_transforms(image)
    return imagetensor

checkpoint = torch.load("catvdog.pt", map_location=torch.device('cpu'))
model = models.densenet121(pretrained=False)
model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))
model.parameters = checkpoint['parameters']
model.load_state_dict(checkpoint['state_dict'])
model.eval()

graham_img_path = "graham2.jpg"
bronte_img_path = "bronte.jpg"
paisley_img_path = "paisley.jpg"

image = image_transformation(paisley_img_path)
image1 = image[None, :, :, :]
prediction = torch.exp(model(image1))
topconf, topclass = prediction.topk(1, dim=1)

if topclass.item() == 1:
    print({'class': 'dog', 'confidence': str(topconf.item())})
else:
    print({'class': 'cat', 'confidence': str(topconf.item())})