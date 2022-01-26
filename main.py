import os
from ARI.Linemod import Linemod
from ARI.ARICOCO import ARICOCO

if __name__ == "__main__":
    sourceDir = "E:\code\PoseFuse\data\Linemod_preprocessed"
    linemod = Linemod(sourceDir, "linemod")
    coco = ARICOCO(linemod)

    coco.visbbox(os.path.join(sourceDir, "data\\02\\rgb"), 20000)
