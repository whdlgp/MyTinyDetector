from pathlib import Path
import math
import numpy as np
import yaml
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw

from block.model import Detector, decode
from dataloader.pascalvoc_trainval import VOC_CLASSES


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_image(image_path, img_size):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((img_size, img_size))
    image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    image = (image - MEAN) / STD
    return image.unsqueeze(0)


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Detector(num_classes=len(VOC_CLASSES)).to(device)
    ckpt = torch.load(Path(config["checkpoint_dir"]) / config["test_checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    image_paths = sorted(Path("data/testset").glob("*.[jJ][pP][gG]"))[:config["test_num"]]

    out_dir = Path(config["test_out"])
    out_dir.mkdir(exist_ok=True)

    images = []

    with torch.no_grad():
        for image_path in image_paths:
            input_image = load_image(image_path, config["img_size"]).to(device)

            output = model(input_image)
            result = decode(output, conf_thresh=config["conf_th"], nms_thresh=config["nms_th"])[0]

            denorm_img = (input_image[0].cpu() * STD + MEAN).clamp(0, 1)
            img = to_pil_image(denorm_img)
            draw = ImageDraw.Draw(img)

            for box, obj, cls_id, score in zip(result["boxes"], result["objectness"], result["class_id"], result["class_score"]):
                xmin, ymin, xmax, ymax = (box * config["img_size"]).tolist()
                label = f"{VOC_CLASSES[cls_id.item()]} obj:{obj.item():.2f} cls:{score.item():.2f}"
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
                draw.text((xmin, ymin), label, fill="red")

            images.append(img)

    grid_size = math.ceil(math.sqrt(len(images)))
    cols = grid_size
    rows = math.ceil(len(images) / cols)
    grid = Image.new("RGB", (config["img_size"] * cols, config["img_size"] * rows))

    for i, img in enumerate(images):
        x = (i % cols) * config["img_size"]
        y = (i // cols) * config["img_size"]
        grid.paste(img, (x, y))

    grid.save(out_dir / "result_grid.png")

    print(f"Done: {out_dir}")