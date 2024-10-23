import os
import json
import pickle
import nltk
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import pytorch_lightning as pl

from model import HybridModel
from vocabulary import Vocabulary

# na, ez mi a fene???
nltk.download('punkt')


class CocoDataset(data.Dataset):
    def __init__(self, data_path, json_path, vocabulary, transform=None):
        self.image_dir = data_path
        self.vocabulary = vocabulary
        self.transform = transform
        with open(json_path) as json_file:
            self.coco = json.load(json_file)
        self.image_id_file_name = dict()
        for image in self.coco['images']:
            self.image_id_file_name[image['id']] = image['file_name']

    def __getitem__(self, idx):
        annotation = self.coco['annotations'][idx]
        caption = annotation['caption']
        tkns = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocabulary('<start>'))
        caption.extend([self.vocabulary(tkn) for tkn in tkns])
        caption.append(self.vocabulary('<end>'))

        image_id = annotation['image_id']
        image_file = self.image_id_file_name[image_id]
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, torch.Tensor(caption)

    def __len__(self):
        return len(self.coco['annotations'])


def coco_collate_fn(data_batch):
    data_batch.sort(key=lambda d: len(d[1]), reverse=True)
    imgs, caps = zip(*data_batch)

    imgs = torch.stack(imgs, 0)

    cap_lens = [len(cap) for cap in caps]
    padded_caps = torch.zeros(len(caps), max(cap_lens)).long()
    for i, cap in enumerate(caps):
        end = cap_lens[i]
        padded_caps[i, :end] = cap[:end]
    return imgs, padded_caps, cap_lens


def get_loader(data_path, json_path, vocabulary, transform, batch_size, shuffle, num_workers=0):
    coco_ds = CocoDataset(data_path=data_path,
                          json_path=json_path,
                          vocabulary=vocabulary,
                          transform=transform)
    coco_dl = data.DataLoader(dataset=coco_ds,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=coco_collate_fn)
    return coco_dl


transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                          (0.229, 0.224, 0.225))])

with open('coco_data/vocabulary.pkl', 'rb') as f:
    vocabulary = pickle.load(f)

coco_data_loader = get_loader('coco_data/images',
                              'coco_data/captions.json',
                              vocabulary,
                              transform,
                              128,
                              shuffle=True,
                              num_workers=4)

hybrid_model = HybridModel(256, 256, 512,
                           len(vocabulary), 1)
trainer = pl.Trainer(max_epochs=5)
trainer.fit(hybrid_model, coco_data_loader)