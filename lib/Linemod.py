import os

from django.forms import models

from lib.ARIAnnotation import ARIAnnotation2D, ARIDataset, ARIImage
from utils.TAUUtils import *

class ARILinemod(ARIDataset):
    intrinsic = np.array([[572.4114, 0., 325.2611],
                          [0., 573.57043, 242.04899],
                          [0., 0., 1.]])

    def __init__(self, sourceDir, name) -> None:
        super().__init__(name)
        self.sourceDir = sourceDir
        self.keyPoint3D = {}
        self.parsekeyPoint3D()
        self.parseDataset(2)
        

    def gtFilePath(self, clsId):
        gtFilePath = os.path.join(self.sourceDir, "data/{:02d}/gt.yml".format(clsId))
        return gtFilePath

    def parseToStandardAnno(self, imageAnno:list):
        imageInfo = ARIImage()
        for item in imageAnno:
            standardAnno = ARIAnnotation2D()
            standardAnno.cls = item['obj_id']
            standardAnno.bbox = item['obj_bb']
            standardAnno.pose = transRT2Matrix4X4(item['cam_R_m2c'], item['cam_t_m2c'])
            standardAnno.keyPoint2D = calculateKeyPoint2D(
                self.keyPoint3D[standardAnno.cls],
                self.intrinsic,
                standardAnno.pose)
            standardAnno.segment = []

            self.clses.add(item['obj_id'])
            
            imageInfo.annotations.append(standardAnno)
        return imageInfo

    def parseDataset(self, clsId):
        annos = load_yaml(self.gtFilePath(clsId))
        for imageId in annos.keys():
            imageInfo = self.parseToStandardAnno(annos[imageId])
            imageInfo.filePath = os.path.join(self.sourceDir, 
                                        "data\{:02d}\\rgb\{:04d}.png".format(clsId, imageId))
            self.images.append(imageInfo)

    def parsekeyPoint3D(self):
        modelsInfo = load_yaml(os.path.join(self.sourceDir, "models\models_info.yml"))
        
        for cls in modelsInfo.keys():
            modelInfo = modelsInfo[cls]
            absX = abs(modelInfo['min_x'])
            absY = abs(modelInfo['min_y'])
            absZ = abs(modelInfo['min_z'])
            ## 俯视，起点第一象限
            self.keyPoint3D[cls] = [
                absX, absY, absZ,
                -absX, absY, absZ,
                -absX, -absY, absZ,
                absX, -absY, absZ,
                absX, absY, -absZ,
                -absX, absY, -absZ,
                -absX, -absY, -absZ,
                absX, -absY, -absZ,
                ]


