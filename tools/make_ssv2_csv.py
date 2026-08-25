"""Build VideoDataset manifests for the local Something-Something v2 copy."""

import json
import re
from pathlib import Path


DATASET_ROOT = Path("dataset/something_something_2")
VIDEO_ROOT = DATASET_ROOT / "20bn-something-something-v2"
LABEL_ROOT = DATASET_ROOT / "labels"

LABEL_FILE = LABEL_ROOT / "labels.json"
TRAIN_FILE = LABEL_ROOT / "train.json"
VAL_FILE = LABEL_ROOT / "validation.json"
TRAIN_OUTPUT = Path("dataset/ssv2_train_paths.csv")
VAL_OUTPUT = Path("dataset/ssv2_val_paths.csv")


def normalize_template(template: str) -> str:
    return re.sub(r"\[([^]]+)\]", r"\1", template)


def load_label_map(path: Path) -> dict[str, int]:
    raw = json.loads(path.read_text())
    if all(str(value).isdigit() for value in raw.values()):
        return {normalize_template(str(key)): int(value) for key, value in raw.items()}
    if all(str(key).isdigit() for key in raw):
        return {normalize_template(str(value)): int(key) for key, value in raw.items()}
    raise ValueError(f"Unsupported label mapping format: {path}")


def build_video_index() -> dict[str, Path]:
    index = {}
    for path in VIDEO_ROOT.iterdir():
        if path.is_file():
            index[path.stem] = path
            index[path.name] = path
    return index


def write_manifest(annotation_path: Path, output_path: Path, label_map: dict[str, int], video_index: dict[str, Path]):
    annotations = json.loads(annotation_path.read_text())
    missing = []
    written = 0

    with output_path.open("w") as output:
        for item in annotations:
            video_id = str(item["id"])
            template = normalize_template(item.get("template", item.get("label")))
            video_path = video_index.get(video_id)
            if video_path is None:
                missing.append(video_id)
                continue
            if template not in label_map:
                raise KeyError(f"Unknown SSv2 template: {template}")
            output.write(f"{video_path.as_posix()} {label_map[template]}\n")
            written += 1

    print(f"{output_path}: wrote {written} entries; missing {len(missing)} videos")
    if missing:
        print(f"First missing IDs: {missing[:10]}")


def main():
    required = [VIDEO_ROOT, LABEL_FILE, TRAIN_FILE, VAL_FILE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths: " + ", ".join(missing))

    label_map = load_label_map(LABEL_FILE)
    video_index = build_video_index()
    print(f"Indexed {len(video_index)} video names")
    write_manifest(TRAIN_FILE, TRAIN_OUTPUT, label_map, video_index)
    write_manifest(VAL_FILE, VAL_OUTPUT, label_map, video_index)


if __name__ == "__main__":
    main()
