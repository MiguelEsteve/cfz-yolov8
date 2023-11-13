import os.path
import cv2
import numpy as np
import torch.cuda
from ultralytics.yolo.utils.plotting import Annotator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO, FastSAM

import sys
sys.path.append('C:\\repos\\vct-ml-yolov8')

from src.displayUtils import Yolov8DisplayUtils, Others
from configs import config
from configs import log_conf

LOGGER = log_conf.getLogger(__name__)


class CardBoardObjectDetection:
    def __init__(self):
        self.model = YOLO('data/yolov8n.pt')

    def set_checkpoint(self, checkpoint):
        self.model = YOLO(checkpoint)

    def finetune(self, datataset_yaml: str = None):
        dataset_yaml = 'I:/Datasets/cardboard/Cardboard.v3-cardboard.yolov8/data.yaml' if datataset_yaml is None else \
            datataset_yaml
        if not os.path.exists(dataset_yaml):
            print(f"{dataset_yaml} not found")
        results = self.model.train(data=dataset_yaml, epochs=10)
        return results

    def predict(self, x, checkpoint: str = None):
        """
        checkpoint = 'C:/repos/vct-ml-yolov8/src/data/runs/detect/train8/weights/best.pt' if checkpoint is None else (
            checkpoint)
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f'{checkpoint}')
        self.model = YOLO(checkpoint)
        """
        results = self.model.predict(source=x, save=True)
        return results

    @staticmethod
    def from_result_to_boxes(results):
        return results[0].boxes

    @staticmethod
    def draw_boxes(img_arr, results):
        for result in results:
            for box in result.boxes:
                pt1 = list(box.xyxy[0][:2].cpu().numpy().astype(int))
                pt2 = list(box.xyxy[0][2:4].cpu().numpy().astype(int))
                img_arr = cv2.rectangle(img_arr, pt1, pt2, thickness=2, color=(200, 0, 0))

                position = box.xyxy[0][:2].cpu().numpy().astype(int)
                position += np.array([3, 15])
                score = box.conf.item()
                class_idx = int(box.cls.item())

                text_to_draw = f'{results[0].names[class_idx]}, {score:.2f}'
                text_size, _ = cv2.getTextSize(text_to_draw, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
                                               thickness=1)
                text_w, text_h = text_size
                x, y = position

                cv2.rectangle(img_arr,
                              pt1=position,
                              pt2=(x + text_w, y + text_h + 8),
                              color=(255, 255, 255), thickness=-1)

                cv2.putText(img_arr,
                            text=f'{results[0].names[class_idx]}, {score:.2f}',
                            org=(x, y + text_h),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.3,
                            color=(0, 0, 255),
                            thickness=1)
        return img_arr

    def apply_to_video(self, video_fn, checkpoint: str = None):
        checkpoint = 'C:/repos/vct-ml-yolov8/src/data/runs/detect/train8/weights/best.pt' if checkpoint is None else (
            checkpoint)
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f'{checkpoint}')
        self.model = YOLO(checkpoint)

        cap = cv2.VideoCapture(video_fn)

        while True:
            _, img = cap.read()
            results = self.model.predict(img)

            for r in results:
                annotator = Annotator(img)
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]
                    c = box.cls
                    annotator.box_label(b, self.model.names[int(c)])
            img = annotator.result()

            cv2.imshow('YOLO v8 Detection', img)
            if cv2.waitKey(25) & 0xFF == ord('S'):
                break
        cap.release()
        cv2.destroyAllWindows()


class CardBoardInstanceSegmentation:
    def __init__(self, pretrained: str = None, level: str = 'DEBUG'):

        pretrained = 'yolov8x-seg.pt' if pretrained is None else pretrained
        self.default_checkpoint = os.path.join(config.PRETRAINED, f'is/{pretrained}')

        self.model = YOLO(self.default_checkpoint)

        LOGGER.setLevel(level)
        LOGGER.debug(f'{self.__class__.__name__} instantiated')

    def set_checkpoint(self, checkpoint):
        self.default_checkpoint = checkpoint
        self.model = YOLO(checkpoint)

    def finetune(self, datataset_yaml: str = None, output_path: str = None):
        dataset_yaml = os.path.join(config.DATASETS_PATH,
                                    'miguelcajas.v4i.yolov8/data.yaml') if datataset_yaml is None else \
            datataset_yaml
        if not os.path.exists(dataset_yaml):
            print(f"{dataset_yaml} not found")

        results = self.model.train(data=dataset_yaml,
                                   epochs=1,
                                   batch=4,
                                   verbose=True,
                                   project=config.DATASETS_PATH)
        return results

    def predict(self, x, checkpoint: str = None):
        checkpoint = self.default_checkpoint if checkpoint is None else checkpoint
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f'{checkpoint}')
        self.model = YOLO(checkpoint)
        results = self.model.predict(source=x,
                                     save=True,
                                     project=config.DATASETS_PATH)
        return results

    @staticmethod
    def from_results_to_bboxes(results):
        return results[0].boxes.cpu().numpy()

    @staticmethod
    def from_results_to_masks(results):
        return results[0].masks.cpu().numpy()

    def draw_masks(self, imgarr, results, return_mask=False):
        if results[0].masks is None:
            if return_mask:
                return imgarr, np.zeros_like(imgarr)
            return imgarr

        imgarr_c = imgarr.copy()
        d = Yolov8DisplayUtils(img_arr=imgarr_c, results=self.from_results_to_masks(results))
        image, mask = d.plot_to_result(annotations=self.from_results_to_masks(results))
        if return_mask:
            return image, mask
        return image

    @staticmethod
    def draw_boxes(imgarr, results):
        if len(results[0].boxes.cpu().numpy()):
            imgarr_c = imgarr.copy()
            image = CardBoardObjectDetection.draw_boxes(img_arr=imgarr_c, results=results)
        else:
            return imgarr
        return image

    def draw_all(self, imgarr, results, resize_factor=0.5):
        image_masked, mask = self.draw_masks(imgarr, results, return_mask=True)
        image_boxes = self.draw_boxes(imgarr, results)
        all_draw = np.concatenate((imgarr, image_boxes, image_masked, mask), axis=1)
        size_ = all_draw.shape[:2]
        all_draw = cv2.resize(all_draw, dsize=(int(size_[1] * resize_factor), int(size_[0] * resize_factor)))

        return all_draw

    def apply_to_video(self, video_fn, checkpoint: str = None):
        checkpoint = self.default_checkpoint if checkpoint is None else checkpoint
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f'{checkpoint}')
        self.model = YOLO(checkpoint)

        from ultralytics.yolo.utils.plotting import Annotator
        cap = cv2.VideoCapture(video_fn)

        count = 0
        while True:
            _, img = cap.read()
            results = self.model.predict(img)
            img = self.draw_all(img, results)

            img = Others.draw_frame_number(img_arr=img, frame_number=count)
            cv2.imshow('YOLO v8 Detection', img)
            count += 1
            if cv2.waitKey(1) & 0xFF == ord('S'):
                break
        cap.release()
        cv2.destroyAllWindows()


class CardBoardFastSAM(CardBoardInstanceSegmentation):
    def __init__(self, pretrained: str = None, level: str = 'INFO'):
        super(CardBoardFastSAM, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pretrained = 'FastSAM-x.pt' if pretrained is None else pretrained
        self.default_checkpoint = os.path.join(config.PRETRAINED, f'fastsam/{pretrained}')

        self.model = FastSAM(self.default_checkpoint)
        self.model.to(self.device)

        self.ins = CardBoardInstanceSegmentation()

        LOGGER.setLevel(level)
        LOGGER.debug(f'{self.__class__.__name__} instantiated')

    def predict(self, x, *args):
        preds = self.model(x)
        return preds

    def eval(self, dataset):
        raise NotImplementedError   # not implemented by ultralytics


class All:
    def __init__(self):
        self.y_default = CardBoardInstanceSegmentation()
        self.y_finetuned = CardBoardInstanceSegmentation()
        self.y_finetuned.set_checkpoint(
            checkpoint=os.path.join(config.PROJECT_PATH, 'tests/runs/segment/train6/weights/best.pt')
        )
        self.fastsam = CardBoardFastSAM()

    def predict(self, x):
        y_results = self.y_default.predict(x)
        y_finetuned_results = self.y_finetuned.predict(x)
        fastsam_results = self.fastsam.predict(x)
        return y_results, y_finetuned_results, fastsam_results

    @staticmethod
    def from_results_to_bboxes(results):
        return results[0].boxes.cpu().numpy()

    @staticmethod
    def from_results_to_masks(results):
        return results[0].masks.cpu().numpy()

    def draw_masks(self, imgarr, results, return_mask=False):
        if results[0].masks is None:
            if return_mask:
                return imgarr, np.zeros_like(imgarr)
            return imgarr

        imgarr_c = imgarr.copy()
        d = Yolov8DisplayUtils(img_arr=imgarr_c, results=self.from_results_to_masks(results))
        image, mask = d.plot_to_result(annotations=self.from_results_to_masks(results))
        if return_mask:
            return image, mask
        return image

    @staticmethod
    def draw_boxes(imgarr, results):
        if len(results[0].boxes.cpu().numpy()):
            imgarr_c = imgarr.copy()
            image = CardBoardObjectDetection.draw_boxes(img_arr=imgarr_c, results=results)
        else:
            return imgarr
        return image

    def _draw(self, imgarr, results, resize_factor=0.5):
        image_masked, mask = self.draw_masks(imgarr, results, return_mask=True)
        image_boxes = self.draw_boxes(imgarr, results)
        all_draw = np.concatenate((imgarr, image_boxes, image_masked, mask), axis=1)
        size_ = all_draw.shape[:2]
        all_draw = cv2.resize(all_draw, dsize=(int(size_[1] * resize_factor), int(size_[0] * resize_factor)))

        return all_draw

    def draw_all(self, imgarr, results: list):
        draws = []
        for result in results:
            draws.append(self._draw(imgarr, result))
        final = np.concatenate(draws, axis=0)
        return final

    def apply_to_video(self, video_fn):
        cap = cv2.VideoCapture(video_fn)

        count = 0
        while True:
            _, img = cap.read()
            results = self.predict(img)
            img = self.draw_all(img, results)

            img = Others.draw_frame_number(img_arr=img, frame_number=count)
            cv2.imshow('YOLO v8 Detection', img)
            count += 1
            if cv2.waitKey(1) & 0xFF == ord('S'):
                break
        cap.release()
        cv2.destroyAllWindows()