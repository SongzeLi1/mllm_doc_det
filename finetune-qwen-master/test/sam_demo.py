import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    image = Image.open("./resource/test/cars.jpg")
    image = np.array(image.convert("RGB"))
    predictor.set_image(image)

    input_box = np.array(
        [
            [75, 275, 1725, 850],
            [425, 600, 700, 875],
            [1375, 550, 1650, 800],
            [1375, 550, 234, 1235],
        ]
    )
    input_point = np.array([[575, 750]])
    input_label = np.array([0])

    masks, scores, logits = predictor.predict(box=None, multimask_output=False)
    # masks, _, _ = predictor.predict(box=np.array[1, 2, 3, 4], multimask_output=False)
