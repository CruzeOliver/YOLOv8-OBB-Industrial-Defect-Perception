from pathlib import Path
import shutil

from ultralytics import YOLO


# Only change values in this section when trying another training setup.
config = {
    "model": "yolov8n-seg.pt",
    "epochs": 150,
    "imgsz": 640,
    "batch": 16,  # Set -1 for automatic GPU batch sizing.
    "device": 0,
    "workers": 2,
    "optimizer": "AdamW",  # SGD, Adam, AdamW, or auto.
    "lr0": 0.001,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "patience": 30,
    "cos_lr": True,
    "degrees": 5.0,
    "translate": 0.05,
    "scale": 0.30,
    "mosaic": 0.50,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "close_mosaic": 10,
    "project": "/kaggle/working/runs",
    "name": "build_seg_custom",
}


def find_dataset() -> Path:
    yaml_files = list(Path("/kaggle/input").rglob("data.yaml"))
    for yaml_file in yaml_files:
        root = yaml_file.parent
        if (root / "images").is_dir() and (root / "labels").is_dir() and (root / "classes.txt").is_file():
            return root
    raise FileNotFoundError("Could not find a Kaggle dataset containing data.yaml, images, labels, and classes.txt")


def prepare_runtime_dataset() -> Path:
    source = find_dataset()
    root = Path("/kaggle/working/dataset")
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(source, root)
    for cache_file in root.rglob("*.cache"):
        cache_file.unlink()
    print("Dataset:", root)
    for split in ("train", "val", "test"):
        image_count = len(list((root / "images" / split).glob("*")))
        label_count = len(list((root / "labels" / split).glob("*.txt")))
        print(f"{split}: images={image_count}, labels={label_count}")
    return root


dataset_root = prepare_runtime_dataset()
model = YOLO(config["model"])
train_args = {key: value for key, value in config.items() if key not in {"model", "project", "name"}}
train_args.update(
    data=str(dataset_root / "data.yaml"),
    project=config["project"],
    name=config["name"],
    amp=True,
)
model.train(**train_args)

best_weights = Path(config["project"]) / config["name"] / "weights" / "best.pt"
metrics = model.val(data=str(dataset_root / "data.yaml"), imgsz=config["imgsz"], split="test")
print("Test metrics:", metrics)

if best_weights.is_file():
    archive_path = Path("/kaggle/working") / f"{config['name']}_weights"
    shutil.make_archive(str(archive_path), "zip", best_weights.parent)
    print("Weights:", f"{archive_path}.zip")
