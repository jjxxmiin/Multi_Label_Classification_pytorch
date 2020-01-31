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

        self.img_path = './datasets/voc/train/VOC{}/JPEGImages'.format(year)
        self.ann_path = './datasets/voc/train/VOC{}/Annotations'.format(year)
        self.spl_path = './datasets/voc/train/VOC{}/ImageSets/Main'.format(year)

        self.train_path = './datasets/voc/train/VOC{}'.format(year)
        self.test_path = './datasets/voc/test/VOC{}'.format(year)

    def get_loader(self, transformer, datatype):
        if datatype == 'train' or datatype == 'val' or datatype == 'trainval':
            path = self.train_path
        elif datatype == 'test':
            path = self.test_path
        else:
            AssertionError("[ERROR] Invalid path")

        custom_voc = VocDataset(path,
                                dataType=datatype,
                                transformer=transformer)

        custom_loader = torch.utils.data.DataLoader(
            dataset=custom_voc,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate)

        return custom_loader
