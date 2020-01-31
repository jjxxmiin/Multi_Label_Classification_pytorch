import os
import sys
import numpy as np
import torch.utils.data as data
from scipy.misc import imread
from PIL import Image

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)


def scaling(data):
    min_value = min(data)
    max_value = max(data)

    for i, x in enumerate(data):
        data[i] = (x - min_value) / max_value - min_value

    return data


class VocDataset(data.Dataset):
    def __init__(self,
                 path,
                 dataType='trainval',
                 transformer=None):
        '''
        :param
            transform: augmentation lib : [img], custom : [img, target]
            target_transform: augmentation [target]
        '''
        type_file = dataType + '.txt'

        with open(os.path.join(os.path.join(path, 'ImageSets/Main'), type_file), 'r') as f:
            file_names = [x.strip() for x in f.readlines()]

        self.imgs = [os.path.join(os.path.join(path, 'JPEGImages'), x + '.jpg') for x in file_names]
        self.anns = [os.path.join(os.path.join(path, 'Annotations'), x + '.xml') for x in file_names]
        self.transformer = transformer
        
        assert (len(self.imgs) == len(self.anns))

    def __getitem__(self, index):
        '''
        :param
            index : index

        :return
            img : (numpy Image)
            target : (numpy) [class_id]
        '''
        img = imread(self.imgs[index], mode='RGB')
        img = Image.fromarray(img)

        target = self.parse_voc(ET.parse(open(self.anns[index])).getroot())

        img = self.transformer(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def parse_voc(self, xml):
        '''
        :param
            xml_path : xml root
        :return
            res : (numpy) [c_id, c_id, ...]
        '''

        objects = xml.findall("object")

        res = np.zeros(len(VOC_CLASSES))

        for object in objects:
            c = object.find("name").text.lower().strip()
            c_id = VOC_CLASSES.index(c)

            res[c_id] = 1

        #scaling_res = scaling(res)
        return res
