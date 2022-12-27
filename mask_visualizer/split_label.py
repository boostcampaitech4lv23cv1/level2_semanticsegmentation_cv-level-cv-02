import os
import csv
import json

from tqdm import tqdm

#load json file
with open("/opt/ml/input/data/train_all.json", "r") as f:
    data = json.load(f)

#data를 저장할 폴더 생성
dir_path = f"/opt/ml/mask_visualizer/data"
if not os.path.isdir(dir_path) :
    os.mkdir(dir_path)

#각 data별 정보 저장
info_data = data['info']
licenses_data = data['licenses']
images_data = data['images']
categories_data = data['categories']
annotations_data = data['annotations']

#label list
label_list = {
    1: "General_trash",
    2: "Paper",
    3: "Paper_pack",
    4: "Metal",
    5: "Glass",
    6: "Plastic",
    7: "Styrofoam",
    8: "Plastic_bag",
    9: "Battery",
    10: "Clothing",
}

for category in list(label_list.keys()) :

    print(f"{label_list[category]}...")

    #file이 저장될 directory 생성
    label_name = label_list[category]
    data_path = dir_path + f"/{category}_{label_name}"
    json_path = data_path + f"/{category}_{label_list[category]}.json"

    json_object = { 
                    "info" : info_data,
                    "licenses" : licenses_data,
                    "images" : [],
                    "categories" : categories_data,
                    "annotations" : []
                }

    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    if not os.path.isdir(f"{data_path}/batch_01_vt") :
        os.mkdir(data_path + "/batch_01_vt")
        os.mkdir(data_path + "/batch_02_vt")
        os.mkdir(data_path + "/batch_03")

    #해당 label이 있는 image id를 저장할 list
    label_data = []
    for i, images in enumerate(annotations_data) :
        if images['category_id'] == category :
            if images['image_id'] not in label_data :
                label_data.append(images['image_id'])

    #image name을 저장할 list
    image_name = [["image_name"],]

    for i, image in enumerate(images_data):

        id = image['id']
        #label_data에 id가 없는 경우 건너뛰기
        if id not in label_data :
            continue

        name = image['file_name']
        name = name.split('a/')
        name = name[-1]
        image_name.append([name])

        image['file_name'] = f'/opt/ml/input/data/{name}'

        name = name.split('.')
        name = name[0]

        name_ = os.path.basename(image['file_name'])
        name_ = name_.split('.')
        name_ = name_[0]

        annotations = list(filter(lambda x: x["image_id"] == id, annotations_data))

        json_object['images'] = [image]
        json_object['annotations'] = annotations

        #개별 json file으로 저장
        json_path = f"{data_path}/{name}.json"
        with open(json_path, "w") as f :
            json.dump(json_object, f, indent=4)
    
    csv_path = data_path + f"/_{label_name}.csv"
    with open(csv_path, 'w') as f :
        writer = csv.writer(f)
        writer.writerows(image_name)