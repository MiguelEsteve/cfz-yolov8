import os

import numpy as np
import cv2

from configs import config


a = np.clip(np.random.rand(640, 640, 3)*255.0, 0, 255).astype(np.uint8)
image_o = os.path.join(config.TEST_IMAGES, 'nothing.jpg')
cv2.imwrite(image_o, a)

