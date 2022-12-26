import os
import json

from tqdm import tqdm

#load json file
with open("/opt/ml/input/data/train_test.json", "r") as f:
    data = json.load(f)

#각 data별 정보 저장
info_data = data['info']
licenses_data = data['licenses']
images_data = data['images']
categories_data = data['categories']
annodataions_data = data['annotations']

#
for i, image in enumerate(tqdm(images_data)):
    id = image['id']

    name = os.path.basename(image['file_name'])
    image['file_name'] = f'../images/{name}'

    name = name.split('.')
    name = name[0]

    #시간복잡도를 줄일 수 있는 방법은 없을까?
    annodataions = list(filter(lambda x: x["id"] == id, annodataions_data))

    #json file에 저장될 dict
    json_object = { 
                    "info" : info_data,
                    "licenses" : licenses_data,
                    "images" : image,
                    "categorise" : categories_data,
                    "annodataions" : annodataions
                }

    #json file으로 저장
    with open(f'anno/{name}.json', "w") as f :
        json.dump(json_object, f, indent=4)