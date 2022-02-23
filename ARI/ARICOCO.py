import os
import cv2
import json
import random
import shutil
import numpy as np
from ARI.ARIAnnotation import ARIDataset
from ARI.ARIUtils import draw2DBBOX


class ARICOCO(object):

    def __init__(self, dataset: ARIDataset, train_ratio=0.7) -> None:
        super().__init__()
        self.train_ratio = train_ratio
        self.dataset = dataset
        self.split()
        self.jsonData = self.json()

    def split(self):
        num_imgs = len(self.dataset.images)
        numImgs_train = int(num_imgs*self.train_ratio)
        imgIds = list(range(num_imgs))
        random.shuffle(imgIds)
        self.trainImgIds = imgIds[:numImgs_train]

    def info(self):
        return {
            "description":"This is stable 1.0 version of the 2014 MS COCO dataset.",
            "url":"http:\/\/mscoco.org",
            "version":"1.0",
            "year":2014,
            "contributor":"Microsoft COCO group",
            "date_created":"2015-01-27 09:11:52.357475"
        }

    def licenses(self):
        return [{
            "url":"http:\/\/creativecommons.org\/licenses\/by-nc-sa\/2.0\/",
            "id":1,
            "name":"Attribution-NonCommercial-ShareAlike License"
        }]

    def images(self):
        images_train = []
        images_val = []
        for id, image in enumerate(self.dataset.images):
            if id in self.trainImgIds:
                images_train.append({
                    "license": 0,
                    "file_name": "{:04d}.png".format(id),
                    "coco_url": "",
                    "height": 480,
                    "width": 640,
                    "date_captured": "",
                    "flickr_url": "",
                    "id": id
                })
            else:
                images_val.append({
                    "license": 0,
                    "file_name": "{:04d}.png".format(id),
                    "coco_url": "",
                    "height": 480,
                    "width": 640,
                    "date_captured": "",
                    "flickr_url": "",
                    "id": id
                })
        return images_train, images_val

    def annotations(self):
        annotations_train = []
        annotations_val = []
        numberOfAnno = 0
        for id, image in enumerate(self.dataset.images):
            if id in self.trainImgIds:
                for anno in image.annotations:
                    annotations_train.append({
                        "id": numberOfAnno,
                        "image_id": id,
                        "category_id": self.queryIDWithClsID(anno.cls),
                        "segmentation": [0, 0],
                        "area": anno.bbox[2] * anno.bbox[3],
                        "bbox": anno.bbox, # [x, y, width, height]
                        "iscrowd": 1,
                    })
                    numberOfAnno = numberOfAnno + 1
            else:
                for anno in image.annotations:
                    annotations_val.append({
                        "id": numberOfAnno,
                        "image_id": id,
                        "category_id": self.queryIDWithClsID(anno.cls),
                        "segmentation": [0, 0],
                        "area": anno.bbox[2] * anno.bbox[3],
                        "bbox": anno.bbox, # [x, y, width, height]
                        "iscrowd": 1,
                    })
                    numberOfAnno = numberOfAnno + 1
        return annotations_train, annotations_val

    def categories(self):
        categories = []
        for id, clsID in enumerate(self.dataset.clses):
            categories.append({
                "supercategory": "none",
                "id": id,
                "name": clsID
            })
        return categories

    def queryIDWithClsID(self, clsID):
        for item in self.categories():
            if item['name'] == clsID:
                return item['id']

    def json(self):
        images = self.images()
        annos = self.annotations()
        json_train = {
            "info": self.info(),
            "licenses": self.licenses(),
            "images": images[0],
            "annotations": annos[0],
            "categories": self.categories()
        }
        json_val = {
            "info": self.info(),
            "licenses": self.licenses(),
            "images": images[1],
            "annotations": annos[1],
            "categories": self.categories()
        }
        return json_train, json_val

    def visbbox(self, sourceDir, annoIDs):
        for annoID in annoIDs:
            
            if annoID not in list(self.jsonData[0].keys()):
                continue

            anno = self.jsonData[0]['annotations'][annoID]
            imagePath = os.path.join(sourceDir, "{:04d}.png".format(anno['image_id']))
            image = cv2.imread(imagePath)
            image = draw2DBBOX(image.copy(), anno['bbox'])
            cv2.imshow("ImageWithBBOX2D", image)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def saveCOCO(self, cocoPath):
        annoFile_train = os.path.join(cocoPath, "annotations/instances_train2017.json")
        annoFile_val = os.path.join(cocoPath, "annotations/instances_val2017.json")
        with open(annoFile_train, 'w') as fp:
            json.dump(self.jsonData[0], fp)
        with open(annoFile_val, "w") as fp:
            json.dump(self.jsonData[1], fp)

        trainDir = os.path.join(cocoPath, "train2017")
        valDir = os.path.join(cocoPath, "val2017")
        for i in range(len(self.dataset.images)):
            if i in self.trainImgIds:
                shutil.copy(self.dataset.images[i].filePath, trainDir)
            else:
                shutil.copy(self.dataset.images[i].filePath, valDir)     
