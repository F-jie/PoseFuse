from __future__ import annotations
import imp
import os

from lib.ARIAnnotation import ARIDataset
from utils.TAUUtils import load_yaml

class ARILinemod(ARIDataset):

    def __init__(self, sourceDir, name) -> None:
        super().__init__()
        self.sourceDir = sourceDir
        self.name = name
        self.parseDataset()

    def gtFilePath(self, clsId):
        gtFilePath = os.path.join(self.sourceDir, "data/{:02d}/gt.yml".format(clsId))
        return gtFilePath

    def parseToStandardAnno(imageAnno):
        pass

    def parseDataset(self, clsId):
        annos = load_yaml(self.gtFilePath(clsId))
        for imageId in annos.keys():
            





