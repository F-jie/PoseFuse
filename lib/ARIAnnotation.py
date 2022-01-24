

class ARIAnnotation2D(object):

    def __init__(self) -> None:
        super().__init__()
        self.cls = -1
        self.bbox = [] 
        self.segment = []
        self.pose = []
        self.keyPoint = []

class ARIImage(object):

    def __init__(self) -> None:
        super().__init__()
        self.filePath = ""
        self.annotations:list[ARIAnnotation2D] = []

class ARIDataset(object):
    def __init__(self) -> None:
        super().__init__()
        self.clses:set[int] = set()
        self.images:list[ARIImage] = [] 

if __name__ == "__main__":
    annotation = ARIAnnotation()
    print(annotation.cls)
