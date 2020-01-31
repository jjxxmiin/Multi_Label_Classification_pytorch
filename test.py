import torch
from PIL import Image
import torchvision.transforms as transforms
from models import load_model

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

test_transformer = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

model = load_model(MODEL_PATH, train=False)

images = test_transformer(Image.open('cat.jpg')).view(1, 3, 224, 224)
images = images.to(device)

pred = model(images)
print(pred)




