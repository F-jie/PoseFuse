from cv2 import line
from lib.Linemod import ARILinemod

if __name__ == "__main__":
    sourceDir = "E:\code\PoseFuse\data\Linemod_preprocessed"
    linemod = ARILinemod(sourceDir, "linemod")
    linemod.images[1].visBBOX3D()
