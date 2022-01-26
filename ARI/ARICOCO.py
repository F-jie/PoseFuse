import os
import cv2
import json
from ARI.ARIAnnotation import ARIDataset
from TAU.TAUUtils import draw2DBBOX


class ARICOCO(object):

    def __init__(self, dataset: ARIDataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.jsonData = self.json()

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
        images = []
        for id, image in enumerate(self.dataset.images):
            images.append({
                "license": 0,
                "file_name": "{:04d}.png".format(id),
                "coco_url": "",
                "height": 480,
                "width": 640,
                "date_captured": "",
                "flickr_url": "",
                "id": id
            })
        return images

    def annotations(self):
        annotations = []
        numberOfAnno = 0
        for id, image in enumerate(self.dataset.images):
            for anno in image.annotations:
                annotations.append({
                    "id": numberOfAnno,
                    "image_id": id,
                    "category_id": self.queryIDWithClsID(anno.cls),
                    "segmentation": [0, 0],
                    "area": anno.bbox[2] * anno.bbox[3],
                    "bbox": anno.bbox, # [x, y, width, height]
                    "iscrowd": 1,
                })
                numberOfAnno = numberOfAnno + 1
        return annotations

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
        return {
            "info": self.info(),
            "licenses": self.licenses(),
            "images": self.images(),
            "annotations": self.annotations(),
            "categories": self.categories()
        }

    def visbbox(self, sourceDir, annoIDs):
        for annoID in annoIDs:
            numberOfAnno = len(self.jsonData['annotations'])
            assert annoID < numberOfAnno, "annoID shold less than {}".format(numberOfAnno)
            anno = self.jsonData['annotations'][annoID]
            imagePath = os.path.join(sourceDir, "{:04d}.png".format(anno['image_id']))
            image = cv2.imread(imagePath)
            image = draw2DBBOX(image.copy(), anno['bbox'])
            cv2.imshow("ImageWithBBOX2D", image)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def saveCOCO(self, cocoJsonFile):
        with open(cocoJsonFile, 'w') as fp:
            json.dump(self.jsonData, fp)
        
