from cv2 import line
from ARI.Linemod import Linemod

if __name__ == "__main__":
    sourceDir = "E:\code\PoseFuse\data\Linemod_preprocessed"
    linemod = Linemod(sourceDir, "linemod")
    linemod.images[1].visBBOX3D()
