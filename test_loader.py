import matplotlib.pyplot as plt
import matplotlib.patches as patches
 
from dataloader.pascalvoc_trainval import get_dataloader, VOC_CLASSES
 
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def denormalize(img_tensor):
    img = img_tensor.permute(1, 2, 0).numpy()
    img = img * STD + MEAN
    return np.clip(img, 0, 1)

if __name__ == "__main__":
    loader = get_dataloader(root="./data", image_set="train", batch_size=4, shuffle=True)
    #loader = get_dataloader(root="./data", image_set="val", batch_size=4, shuffle=True)
    images, boxes, labels = next(iter(loader))
    
    print("image shape:", images.shape)
    print("value range: min", images.min().item(), "max", images.max().item(), "mean", images.mean().item())
    print("Total data: ", len(loader.dataset))
    
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    
    for i, img in enumerate(images):
        ax = axes[i]
        ax.imshow(denormalize(img))
    
        h, w = img.shape[1], img.shape[2]
        for box, label in zip(boxes[i], labels[i]):
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle(
                (xmin * w, ymin * h), (xmax - xmin) * w, (ymax - ymin) * h,
                linewidth=2, edgecolor="red", facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(xmin * w, ymin * h, VOC_CLASSES[int(label)], color="red")
    
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig("dataloader_check.png")
    print("saved dataloader_check.png")
    