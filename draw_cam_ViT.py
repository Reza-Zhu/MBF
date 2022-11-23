import argparse
import cv2
import numpy as np
import torch
import glob
import re
import os
import Multi_HBP
from einops import rearrange
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def draw_heat_map(weights, img_path):

    count = 0
    for weight in weights:
        print(weight)
        model = Multi_HBP.Hybird_ViT(701, 0)

        model.load_state_dict(torch.load(weight))
        model = model.model_1

        model.eval()

        if args.use_cuda:
            model = model.cuda()
        # model.model_1.
        target_layers = [model.last_block[-1].norm1]

        if args.method not in methods:
            raise Exception(f"Method {args.method} not implemented")

        if args.method == "ablationcam":
            cam = methods[args.method](model=model,
                                       target_layers=target_layers,
                                       use_cuda=args.use_cuda,
                                       reshape_transform=reshape_transform,
                                       ablation_layer=AblationLayerVit())
        else:
            cam = methods[args.method](model=model,
                                       target_layers=target_layers,
                                       use_cuda=args.use_cuda,
                                       reshape_transform=reshape_transform)

        rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (384, 384))
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        print(input_tensor.shape)

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested category.
        targets = [ClassifierOutputTarget(-1)]

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            eigen_smooth=args.eigen_smooth,
                            aug_smooth=args.aug_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam)
        plt.figure("black")
        plt.imshow(cam_image)
        plt.show()

        cv2.imwrite(os.path.join("./draw_imgs", f'{args.method}_cam_%d_vit.jpg' % count), cam_image)
        count += 5


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')

    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='eigengradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=24, width=24):
    # print(tensor.shape)
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    # print(result.shape)
    # result = rearrange(result, "b (h w) y -> b y h w", h=24, w=24)
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    # print(result.shape)

    return result


if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    paths = ["/home/sues/save_model_weight/Release_final_weight/net_077.pth"]

    img_path = "/home/sues/media/disk2/University-Release/University-Release/test/gallery_satellite/0001/0001.jpg"
    # model.load_state_dict(torch.load(weights[0]))
    draw_heat_map(paths, img_path)
