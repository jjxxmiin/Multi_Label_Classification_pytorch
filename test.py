import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

from PIL import Image
from datasets.loader import VOC


if torch.cuda.is_available():
    device = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
    torch.set_default_tensor_type('torch.FloatTensor')

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

MODEL_PATH = 'model.h5'
BATCH_SIZE = 32

# test dataset
test_transformer = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor()])

voc = VOC(batch_size=BATCH_SIZE, year="2007")
test_loader = voc.get_loader(transformer=test_transformer, datatype='test')

# load model
model = models.vgg16().to(device)
model.classifier[6] = nn.Linear(4096, 20)

# load weight
model.load_state_dict(torch.load(MODEL_PATH))

# model eval
model = model.eval()

# tensor image generate
images = test_transformer(Image.open('cat.jpg')).view(1, 3, 224, 224)
images = images.to(device)

# prediction
pred = model(images)

print(VOC_CLASSES[pred.argmax()])