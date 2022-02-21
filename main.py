import os
from Linemod import Linemod
from ARI.ARICOCO import ARICOCO

if __name__ == "__main__":
    sourceDir = "E:\code\PoseFuse\data\Linemod_preprocessed"
    linemod = Linemod(sourceDir, "linemod")
    coco = ARICOCO(linemod)

    coco.visbbox(os.path.join(sourceDir, "data\\02\\rgb"), [190,8990,910,3763])
    coco.saveCOCO(os.path.join(sourceDir, "coco\instances_train2017.json"))
