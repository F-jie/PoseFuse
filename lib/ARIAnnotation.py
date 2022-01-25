import cv2
from utils.TAUUtils import draw2DBBOX, draw3DBBOX 

class ARIAnnotation2D(object):

    def __init__(self) -> None:
        super().__init__()
        self.cls = -1
        self.bbox = [] 
        self.segment = []
        self.pose = []
        self.keyPoint2D = []

class ARIImage(object):

    def __init__(self) -> None:
        super().__init__()
        self.filePath = ""
        self.annotations:list[ARIAnnotation2D] = []

    def visImageAndAnno(self):
        image = cv2.imread(self.filePath)
        cv2.imshow("RowImage", image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def visBBOX2D(self):
        image = cv2.imread(self.filePath)
        for anno in self.annotations:
            bbox2d = anno.bbox
            imageBOX = draw2DBBOX(image.copy(), bbox2d)
            cv2.imshow("ImageWithBBOX2D", imageBOX)
            cv2.waitKey()
            cv2.destroyAllWindows()
    
    def visBBOX3D(self):
        image = cv2.imread(self.filePath)
        for anno in self.annotations:
            bbox3d = anno.keyPoint2D
            imageBOX = draw3DBBOX(image.copy(), bbox3d)
            cv2.imshow("ImageWithBBOX3D", imageBOX)
            cv2.waitKey()
            cv2.destroyAllWindows()


class ARIDataset(object):
    def __init__(self, name) -> None:
        super().__init__()
        self.clses:set[int] = set()
        self.images:list[ARIImage] = [] 
        self.name = name

if __name__ == "__main__":
    annotation = ARIAnnotation2D()
    print(annotation.cls)
