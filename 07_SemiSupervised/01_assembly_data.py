import os
import subprocess
import json
import random
import nltk
import pickle
import shutil as shu
from collections import Counter
from PIL import Image
from vocabulary import Vocabulary


def resize_images(input_path, output_path, new_size):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    image_files = os.listdir(input_path)
    num_images = len(image_files)
    for i, img in enumerate(image_files):
        img_full_path = os.path.join(input_path, img)
        with open(img_full_path, 'r+b') as f:
            with Image.open(f) as image:
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                img_sv_full_path = os.path.join(output_path, img)
                image.save(img_sv_full_path, image.format)
        if (i + 1) % 100 == 0 or (i + 1) == num_images:
            print("Resized {} out of {} total images.".format(i + 1, num_images))


def build_vocabulary(json_path, threshold):
    with open(json_path) as json_file:
        captions = json.load(json_file)
    counter = Counter()
    i = 0
    for annotation in captions['annotations']:
        i = i + 1
        caption = annotation['caption']
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        if i % 1000 == 0 or i == len(captions['annotations']):
            print("Tokenized {} out of total {} captions.".format(i, len(captions['annotations'])))

    tokens = [tkn for tkn, i in counter.items() if i >= threshold]

    vocabulary = Vocabulary()
    vocabulary.add_token('<pad>')
    vocabulary.add_token('<start>')
    vocabulary.add_token('<end>')
    vocabulary.add_token('<unk>')

    for i, token in enumerate(tokens):
        vocabulary.add_token(token)
    return vocabulary

#####################################################
# Főprogram innen indul
#####################################################

obj_fl = "./coco_data/annotations/instances_train2017.json"
with open(obj_fl) as json_file:
    object_detections = json.load(json_file)

CATEGORY_LIST = [4, 5, 22, 43]

COUNT_PER_CATEGORY = 1000

category_dict = dict()
for category_id in CATEGORY_LIST:
    category_dict[category_id] = dict()

all_images = dict()
filtered_images = set()

for annotation in object_detections['annotations']:
    category_id = annotation['category_id']
    image_id = annotation['image_id']
    area = annotation['area']
    if category_id in CATEGORY_LIST:
        if image_id not in category_dict[category_id]:
            category_dict[category_id][image_id] = []
    if image_id not in all_images:
        all_images[image_id] = dict()
    if category_id not in all_images[image_id]:
        all_images[image_id][category_id] = area
    else:
        current_area = all_images[image_id][category_id]
        if area > current_area:
            all_images[image_id][category_id] = area

if COUNT_PER_CATEGORY == -1:
    for category_id in category_dict:
        print("Processing category {}".format(category_id))
        filtered_images.update(category_dict[category_id].keys())
        print("  Filtered total {} images of category {}".format(len(category_dict[category_id].keys()), category_id))
else:
    for image_id in all_images:
        areas = list(all_images[image_id].values())
        categories = list(all_images[image_id].keys())
        sorted_areas = sorted(areas, reverse=True)
        sorted_categories = []
        for area in sorted_areas:
            sorted_categories.append(categories[areas.index(area)])
        all_images[image_id] = sorted_categories

    for category_id in category_dict:
        print("Processing category {}".format(category_id))
        for image_id in category_dict[category_id]:
            category_dict[category_id][image_id] = all_images[image_id]
        prominance_index = 0
        prominent_image_ids = []
        while len(category_dict[category_id]) > 0 and len(prominent_image_ids) < COUNT_PER_CATEGORY:
            remaining_count = COUNT_PER_CATEGORY - len(prominent_image_ids)
            image_ids = []
            for image_id in category_dict[category_id]:
                if category_dict[category_id][image_id].index(category_id) == prominance_index:
                    image_ids.append(image_id)
            for image_id in image_ids:
                del category_dict[category_id][image_id]
            if len(image_ids) <= remaining_count:
                prominent_image_ids = prominent_image_ids + image_ids
                if prominance_index > 4:
                    print(image_ids)
                print("  Added all {} images at prominance_index {}".format(len(image_ids), prominance_index))
            else:
                random.shuffle(image_ids)
                prominent_image_ids = prominent_image_ids + image_ids[0:remaining_count]
                print("  Added {} images at prominance_index {} out of {} images".format(remaining_count, prominance_index, len(image_ids)))
            prominance_index = prominance_index + 1
        filtered_images.update(prominent_image_ids)
        print("  Completed filtering of total {} images of category {}".format(len(prominent_image_ids), category_id))

print("Processed all categories. Number of filtered images is {}".format(len(filtered_images)))


caps_fl = "./coco_data/annotations/captions_train2017.json"
with open(caps_fl) as json_file:
    captions = json.load(json_file)

filtered_annotations = []
for annotation in captions['annotations']:
    if annotation['image_id'] in filtered_images:
        filtered_annotations.append(annotation)
captions['annotations'] = filtered_annotations
print("Number of filtered annotations is {}".format(len(captions['annotations'])))

images = []
filtered_image_file_names = set()
for image in captions['images']:
    if image['id'] in filtered_images:
        images.append(image)
        filtered_image_file_names.add(image['file_name'])
captions['images'] = images
print("Expected number of filtered images is {}, actual number is {}".format(len(filtered_images), len(captions['images'])))

# ezzel itt el fogok bíbelődni
with open("./coco_data/captions.json", 'w+') as output_file:
    json.dump(captions, output_file)

# megoldás colab, linux környezetben:
# !rm -rf ./coco_data/annotations
# windows módszer:
# subprocess.run(["del", "./coco_data/annotations/*.*"])  # "/f", "/s", "/q", , "1>nul"
# subprocess.run(["rmdir", "./coco_data/annotations"])  # "/s", "/q",

# megoldás colab, linux környezetben:
# !mkdir coco_data/images
# windows módszer:
os.mkdir("coco_data/images")

for file_name in filtered_image_file_names:
    shu.copyfile("./coco_data/train2017/{}".format(file_name),
                 "./coco_data/images/{}".format(file_name))

# megoldás colab, linux környezetben:
# !rm -rf ./coco_data/train2017
# windows módszer:
# de nem biztos, hogy egyáltalán ki kéne ezt gyomlálni, ha van hely...
# shu.rmtree("./coco_data/train2017")

nltk.download('punkt')

# ezt is kéri:
nltk.download('punkt_tab')

input_path = './coco_data/images/'
output_path = './coco_data/resized_images/'
new_size = [256, 256]
resize_images(input_path, output_path, new_size)

# megoldás colab, linux környezetben:
# !rm -rf ./coco_data/images
# windows módszer:
# shu.rmtree("./coco_data/images")


# megoldás colab, linux környezetben:
# !mv ./coco_data/resized_images ./coco_data/images
# windows módszer:
# mivel ez a vége, talán jobb, ha marad. lehet, kimaradt kód...
# shu.rmtree("./coco_data/resized_images")


vocabulary = build_vocabulary(json_path='coco_data/captions.json', threshold=4)
vocabulary_path = './coco_data/vocabulary.pkl'
with open(vocabulary_path, 'wb') as f:
    pickle.dump(vocabulary, f)
print("Total vocabulary size: {}".format(len(vocabulary)))