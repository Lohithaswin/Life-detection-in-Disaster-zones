import os
import zipfile
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def download_file(url, dest_path):
    if not dest_path.exists():
        print(f"Downloading {url} to {dest_path}...")
        urllib.request.urlretrieve(url, dest_path)
    return dest_path

def main():
    dataset_dir = Path("datasets/coco_val_subset")
    images_dir = dataset_dir / "images" / "val2017"
    labels_dir = dataset_dir / "labels" / "val2017"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    labels_zip_path = dataset_dir / "coco2017labels.zip"
    labels_url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip"
    download_file(labels_url, labels_zip_path)

    print("Extracting labels...")
    with zipfile.ZipFile(labels_zip_path, 'r') as zip_ref:
        # Only extract the val2017 labels
        for member in zip_ref.namelist():
            if member.startswith("coco/labels/val2017/") and member.endswith(".txt"):
                member_path = Path(member)
                dest = labels_dir / member_path.name
                if not dest.exists():
                    with zip_ref.open(member) as source, open(dest, "wb") as target:
                        target.write(source.read())

    # Get all label files and sort them to be deterministic
    all_label_files = sorted(list(labels_dir.glob("*.txt")))
    subset_labels = all_label_files[:300]
    
    # Delete the others to save space/ensure subset correctness
    for lbl in all_label_files[300:]:
        lbl.unlink()

    # The image ID is the stem of the label file (e.g. 000000397133.txt -> 000000397133.jpg)
    img_ids = [lbl.stem for lbl in subset_labels]

    def download_image(img_id):
        img_filename = f"{img_id}.jpg"
        img_url = f"http://images.cocodataset.org/val2017/{img_filename}"
        img_dest = images_dir / img_filename
        if not img_dest.exists():
            try:
                urllib.request.urlretrieve(img_url, img_dest)
            except Exception as e:
                print(f"Failed to download {img_url}: {e}")

    print(f"Downloading 300 images from val2017...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(download_image, img_ids)

    # Write the YAML config
    yaml_content = f"""path: {dataset_dir.absolute().as_posix()}
train: images/val2017
val: images/val2017

names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
"""
    yaml_path = Path("coco_val_subset.yaml")
    yaml_path.write_text(yaml_content)
    print(f"Created {yaml_path}. Done.")

if __name__ == "__main__":
    main()
