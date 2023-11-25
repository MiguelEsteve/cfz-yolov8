import os.path

import cv2
import numpy as np

import configs.config
from src.cardboards import CardBoardObjectDetection, CardBoardInstanceSegmentation, CardBoardFastSAM, All

import sys

sys.path.append('C:\\repos\\vct-ml-yolov8')
from configs import config


class TestCardboardObjectDetection:
    def __init__(self):
        self.c = CardBoardObjectDetection()

    def test_finetune(self):
        checkpoint = 'C:/repos/vct-ml-yolov8/src/data/runs/detect/train8/weights/best.pt'
        dataset_yaml = 'I:/Datasets/cardboard/miguelcajas.v1i.yolov8/data.yaml'
        self.c.set_checkpoint(checkpoint=checkpoint)
        self.c.finetune(datataset_yaml=dataset_yaml)

    def test_predict(self, plot=True):
        test_image = os.path.join(config.TEST_IMAGES, 'dog.jpeg')
        img_arr = cv2.imread(test_image)
        results = self.c.predict(x=img_arr)
        if plot:
            self.c.draw_boxes(img_arr, results)

    def test_apply_to_video(self):
        checkpoint = '../src/data/yolov8n.pt'  # COCO128 dataset
        checkpoint = 'C:/repos/vct-ml-yolov8/tests/runs/detect/train2/weights/best.pt'
        video_fn = '../vid1.mp4'
        self.c.apply_to_video(video_fn=video_fn, checkpoint=checkpoint)


class TestCardboardInstanceSegmentation:
    def __init__(self):
        self.cis = CardBoardInstanceSegmentation()

    def test_finetune(self):
        self.cis.finetune()

    def test_predict(self):
        test_image = os.path.join(config.TEST_IMAGES, 'nothing.jpg')
        if not os.path.exists(test_image):
            raise FileNotFoundError(f'{test_image}')
        results = self.cis.predict(x=test_image)

        img_arr = cv2.imread(test_image)
        # self.cis.draw_masks(imgarr=img_arr, results=results)
        image = self.cis.draw_all(img_arr, results)
        cv2.imshow("", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_apply_to_video(self):
        test_video = os.path.join(config.TEST_VIDEOS, 'vid1.mp4')
        checkpoint = os.path.join(config.PROJECT_PATH, 'tests/runs/segment/train6/weights/best.pt')
        self.cis.apply_to_video(video_fn=test_video, checkpoint=checkpoint)


class TestCardBoardFastSAM:
    def __init__(self):
        self.t = CardBoardFastSAM()

    def test_predict(self):
        test_image = os.path.join(config.TEST_IMAGES, 'cardboard.jpg')
        if not os.path.exists(test_image):
            raise FileNotFoundError(f'{test_image}')
        imgarr = cv2.imread(test_image)
        results = self.t.predict(imgarr)

        img_arr = cv2.imread(test_image)
        # self.cis.draw_masks(imgarr=img_arr, results=results)
        image = self.t.draw_all(img_arr, results)
        cv2.imshow("", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_eval(self):
        data = configs.config.DATASETS_PATH + '/miguelcajas.v4i.yolov8/data.yaml'
        if not os.path.exists(data):
            raise FileNotFoundError(f'{data}')
        self.t.eval(data)

    def test_apply_to_video(self):
        test_video = os.path.join(config.TEST_VIDEOS, 'vid1.mp4')
        self.t.apply_to_video(video_fn=test_video, checkpoint=self.t.default_checkpoint)


class TestAll:
    def __init__(self):
        self.a = All()

    def test_apply_to_video(self):
        test_video = os.path.join(config.TEST_VIDEOS, 'vid1.mp4')
        self.a.apply_to_video(video_fn=test_video)


if __name__ == '__main__':
    # ------------------------------------------
    # t = TestCardboardObjectDetection()
    # t.test_finetune()
    # t.test_predict()
    # t.test_apply_to_video()

    # ------------------------------------------
    # t = TestCardboardInstanceSegmentation()
    # t.test_finetune()
    # t.test_predict()
    # t.test_apply_to_video()

    # ------------------------------------------
    t = TestCardBoardFastSAM()
    # t.test_predict()
    # t.test_eval()
    t.test_apply_to_video()

    # ------------------------------------------
    # t = TestAll()
    # t.test_apply_to_video()
