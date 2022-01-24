
from lib.ARIAnnotation import ARIDataset


class ARICOCO(object):

    def __init__(self, dataset: ARIDataset) -> None:
        super().__init__()
        self.datasets = dataset

    def info():
        return {
            "description":"This is stable 1.0 version of the 2014 MS COCO dataset.",
            "url":"http:\/\/mscoco.org",
            "version":"1.0",
            "year":2014,
            "contributor":"Microsoft COCO group",
            "date_created":"2015-01-27 09:11:52.357475"
        }

    def licenses():
        return [{
            "url":"http:\/\/creativecommons.org\/licenses\/by-nc-sa\/2.0\/",
            "id":1,
            "name":"Attribution-NonCommercial-ShareAlike License"
        }]

    def images():
        pass

    def annotations():
        pass

    def categories():
        pass

    def json(self):
        return {
            "info": self.info(),
            "licenses": self.licenses(),
            "images": self.images(),
            "annotations": self.annotations(),
            "categories": self.categories()
        }
