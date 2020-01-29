import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models import load_model
import torchvision.transforms as transforms
from datasets.loader import VOC

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)
MODEL_PATH = 'model.h5'
BATCH_SIZE = 16
EPOCH = 200

if torch.cuda.is_available():
    device = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
    torch.set_default_tensor_type('torch.FloatTensor')

# augmentation
train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

valid_transformer = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

voc = VOC(batch_size=BATCH_SIZE, year="2007")
train_loader = voc.get_loader(transformer=train_transformer, type='train')
valid_loader = voc.get_loader(transformer=valid_transformer, type='val')

model = load_model(MODEL_PATH, train=True)

optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                           milestones=[50, 100, 150],
                                           gamma=0.1)
criterion = nn.BCEWithLogitsLoss()

best_loss = 100
train_iter = len(train_loader)
valid_iter = len(valid_loader)

for e in range(EPOCH):
    train_loss = 0
    valid_loss = 0

    scheduler.step()

    for i, (images, targets) in tqdm(enumerate(train_loader), total=train_iter):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        # forward
        pred = model(images)
        # loss
        loss = criterion(pred.double(), targets)
        train_loss += loss.item()
        # backward
        loss.backward(retain_graph=True)
        # weight update
        optimizer.step()

    total_train_loss = train_loss / train_iter

    with torch.no_grad():
        for images, targets in valid_loader:
            images = images.to(device)
            targets = targets.to(device)

            pred = model(images)
            # loss
            loss = criterion(pred.double(), targets)
            valid_loss += loss.item()

    total_valid_loss = valid_loss / valid_iter

    print("[train loss / %f] [valid loss / %f]" % (total_train_loss, total_valid_loss))

    if best_loss > total_valid_loss:
        print("model saved")
        torch.save(model.state_dict(), 'model.h5')
        best_loss = total_valid_loss

"""
from utils import save_tensor_image

for image, targets in train_loader:
    save_tensor_image(image[0])
    print(targets)
    break
"""