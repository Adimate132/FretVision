# renames /data/images/train & .../val to match cvat's label naming conventions. RUN THIS BEFORE TRAINING
from pathlib import Path

def rename_images(split):
    img_dir = Path(f"data/images/{split}")
    lbl_dir = Path(f"data/labels/{split}")

    images = sorted(img_dir.glob("*.jpg"))
    labels = sorted(lbl_dir.glob("*.txt"))

    assert len(images) == len(labels), f"{split}: image/label count mismatch"

    for img, lbl in zip(images, labels):
        new_img = img_dir / f"{lbl.stem}.jpg"
        if new_img.exists():
            raise RuntimeError(f"{split}: {new_img} already exists")
        img.rename(new_img)

    print(f"{split}: renamed {len(images)} images")

rename_images("train")
rename_images("val")
