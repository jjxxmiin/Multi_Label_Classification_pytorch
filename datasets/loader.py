import torch
import torchvision
from datasets.pascal_voc import VocDataset


class CIFAR10(object):
    """
    image shape : 32 x 32
    ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    """

    def __init__(self,
                 batch_size):

        self.classes = 10
        self.batch_size = batch_size

    def get_loader(self, transformer, mode='train', shuffle=True):
        if mode == 'train':
            train = True
        else:
            train = False

        cifar10_dataset = torchvision.datasets.CIFAR10(root='./datasets',
                                                       train=train,
                                                       transform=transformer,
                                                       download=True)

        cifar10_loader = torch.utils.data.DataLoader(cifar10_dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=shuffle)

        return cifar10_loader


def collate(batch):
    '''
    :batch:
    :return:
    images : (tensor)
    targets : (list) [(tensor), (tensor)]
    '''
    targets = []
    images = []

    for x in batch:
        targets.append(torch.from_numpy(x[1]))
        images.append(x[0])

    return torch.stack(images, 0), torch.stack(targets, 0)


class VOC(object):
    def __init__(self,
                 batch_size,
                 year='2007'):

        self.classes = 20
        self.batch_size = batch_size
        self.img_path = './datasets/voc/VOC{}/JPEGImages'.format(year)
        self.ann_path = './datasets/voc/VOC{}/Annotations'.format(year)
        self.spl_path = './datasets/voc/VOC{}/ImageSets/Main'.format(year)

    def get_loader(self, transformer, type):
        custom_voc = VocDataset(
            self.img_path,
            self.ann_path,
            self.spl_path,
            dataType=type,
            transformer=transformer)

        custom_loader = torch.utils.data.DataLoader(
            dataset=custom_voc,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate)

        return custom_loader
