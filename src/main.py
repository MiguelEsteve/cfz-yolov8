import os
import yaml
import numpy as np
import cv2
import supervision as sv
from PIL import Image
from tqdm.auto import tqdm
from ultralytics import YOLO
from datasets import load_dataset
from groundingdino.util.inference import load_model, load_image, predict, annotate

def run_dino(dino, image, text_prompt='food', box_threshold=0.4, text_threshold=0.1):
    boxes, logits, phrases = predict(
        model=dino,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    return boxes, logits, phrases

print(os.getcwd())

model_config = '../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
weights_file = '../GroundingDINO/groundingdino_swint_ogc.pth'
dino = load_model(model_config_path=model_config,
                  model_checkpoint_path=weights_file)
image_source, image = load_image('../dog.jpeg')
boxes, logits, phrases = run_dino(dino, image, text_prompt='dog')
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imshow("", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


def annotate(dino, data, data_size, data_dir):
    data = data.train_test_split(train_size=min(len(data), data_size))['train']

    image_dir = f'{data_dir}/images'
    label_dir = f'{data_dir}/labels'
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for i, d in enumerate(tqdm(data)):
        image_path = f'{image_dir}/{i:06d}.png'
        label_path = f'{label_dir}/{i:06d}.txt'
        image = d['image'].resize((640, 640))
        image.save(image_path)

        image_source, image = load_image(image_path)
        boxes, logits, phrases = run_dino(dino, image)

        label = ['0 ' + ' '.join(list(map(str, b))) for b in boxes.tolist()]
        label = '\n'.join(label)
        with open(label_path, 'w') as f:
            f.write(label)


data = load_dataset('food101')

annotate(dino, data['train'], 3000, 'data/train')
annotate(dino, data['validation'], 1000, 'data/valid')

config = {
    'names': ['food'],
    'nc': 1,
    'train': 'train/images',
    'val': 'valid/images'
}

with open('data/data.yaml', 'w') as f:
    yaml.dump(config, f)