import os
import json
import numpy as np
import pandas as pd
import argparse
import random

from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

path = os.path.dirname(os.path.abspath(__file__))
data_path = "/opt/ml/input/data"

annotations_path = os.path.join(data_path, "train_all.json")


def main(args):
    random.seed(args.random_seed)
    with open(annotations_path) as f:
        data = json.load(f)
        images = data["images"]
        categories = data["categories"]
        annotations = data["annotations"]

    annotations_df = pd.DataFrame.from_dict(annotations)

    var = [(ann["image_id"], ann["category_id"]) for ann in data["annotations"]]

    X = np.ones((len(data["annotations"]), 1))
    y = np.array([v[1] for v in var])
    groups = np.array([v[0] for v in var])

    cv = StratifiedGroupKFold(
        n_splits=args.n_split, shuffle=True, random_state=args.random_seed
    )

    path = args.path

    if not os.path.exists(path):
        os.mkdir(path)

    for idx, (train_index, val_index) in tqdm(enumerate(cv.split(X, y, groups)), total=args.n_split):

        train_dict = dict()
        val_dict = dict()

        for i in ["info", "licenses", "categories"]:
            train_dict[i] = data[i]
            val_dict[i] = data[i]

        train_index = list(set(groups[train_index]))
        val_index = list(set(groups[val_index]))

        train_index.sort()
        val_index.sort()

        train_dict["images"] = np.array(images)[train_index].tolist()
        val_dict["images"] = np.array(images)[val_index].tolist()

        train_dict["annotations"] = annotations_df[
            annotations_df["image_id"].isin(train_index)
        ].to_dict("records")
        val_dict["annotations"] = annotations_df[
            annotations_df["image_id"].isin(val_index)
        ].to_dict("records")

        train_dir = os.path.join(path, f"train_fold{idx}.json")
        val_dir = os.path.join(path, f"val_fold{idx}.json")

        with open(train_dir, "w") as train_file:
            json.dump(train_dict, train_file)

        with open(val_dir, "w") as val_file:
            json.dump(val_dict, val_file)

    print("Done Make files")


def update_dataset(index, mode, input_json, output_dir):

    with open(input_json) as json_reader:
        dataset = json.load(json_reader)

    images = dataset["images"]
    annotations = dataset["annotations"]
    categories = dataset["categories"]

    image_ids = [x.get("id") for x in images]
    image_ids.sort()

    image_ids_train = set(image_ids)

    train_images = [x for x in images if x.get("id") in image_ids_train]

    train_id2id = dict()

    for i in range(len(train_images)):
        train_id2id[train_images[i]["id"]] = i
        train_images[i]["id"] = i

    train_annotations = [x for x in annotations if x.get("image_id") in image_ids_train]

    for i in range(len(train_annotations)):
        train_annotations[i]["image_id"] = train_id2id[train_annotations[i]["image_id"]]

    train_data = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories,
    }

    output_train_json = os.path.join(output_dir, f"{mode}_fold{index}.json")

    print(f"write {output_train_json}")
    with open(output_train_json, "w") as train_writer:
        json.dump(train_data, train_writer)


def loop_n_split(n):
    stratified_path = os.path.join(path, "/opt/ml/input/data", "stratified_group_kfold")
    print("image id's updating...")

    for i in range(n):
        update_dataset(
            index=i,
            mode="train",
            input_json=os.path.join(stratified_path, f"train_fold{i}.json"),
            output_dir=stratified_path,
        )
        update_dataset(
            index=i,
            mode="val",
            input_json=os.path.join(stratified_path, f"val_fold{i}.json"),
            output_dir=stratified_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default=os.path.join(path, "/opt/ml/input/data", "stratified_group_kfold"),
    )
    parser.add_argument("--n_split", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=7)
    args = parser.parse_args()
    main(args)
    loop_n_split(args.n_split)