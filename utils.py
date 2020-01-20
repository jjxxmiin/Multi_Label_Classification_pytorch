import numpy as np
import cv2


def save_tensor_image(image, saved_path='test.png'):
    '''
    :param image: (tensor) cpu image
    :return: (file) save image
    '''

    image = image.permute(1, 2, 0).numpy() * 255.0
    image = image.astype('uint8')

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(saved_path, image)

    print('Finish image save testing')