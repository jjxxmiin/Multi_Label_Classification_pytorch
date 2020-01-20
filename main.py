import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from models import VGG
import torchvision.transforms as transforms
from datasets.loader import VOC
from torch.autograd import Variable
from utils import save_tensor_image

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

batch_size = 16
epoch = 200

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
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

test_transformer = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

voc = VOC(batch_size=batch_size, year="2007")
train_loader = voc.get_loader(transformer=train_transformer, type='train')
valid_loader = voc.get_loader(transformer=train_transformer, type='val')

model = VGG()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                           milestones=[50, 100, 150],
                                           gamma=0.1)
criterion = nn.BCEWithLogitsLoss()

best_loss = 100
train_iter = len(train_loader)
valid_iter = len(valid_loader)

for epoch in range(epoch):
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
for image, targets in train_loader:
    save_tensor_image(image[0])
    print(targets)
    break
"""