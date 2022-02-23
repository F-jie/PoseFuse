import os
from Linemod import Linemod
from ARI.ARICOCO import ARICOCO
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Try use Transforrmer estimate 6DoF Pose", add_help=False)
    parser.add_argument("--use", type=str, default="Linemod2COCO", help="Values: Linemod2COCO, ...")
    args = parser.parse_args()

    if args.use == "Linemod2COCO":
        linemod_root = "/home/hnu/data/Linemod_preprocessed"
        linemod_cls_2_dir = os.path.join(linemod_root, "data/02/rgb")
        linemod = Linemod(linemod_root, "linemod-2")
        coco = ARICOCO(linemod)
        coco.visbbox(linemod_cls_2_dir, [190,8990,910,3763])

        for i, image in enumerate(linemod.images):
            image.visBBOX3D()
            if i == 1:
                break;
        coco.saveCOCO("/home/hnu/Documents/data/")
    else:
        raise ValueError(f"arg {args.use} is not support!")
