import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import os
from unet import Unet
import json
import shutil


def adjustImage(input_img, multiple=32):
    """ Adjusts Dimensions of the Image for inference by padding with zeros along edges.

    Args:
        multiple:
        input_img (ndarray): RGB input.

    Returns:
        ndarray: Padded output.
    """
    new_shape = [None, None, 3]
    img_shape = input_img.shape
    if input_img.shape[0] % multiple == 0:
        new_shape[0] = input_img.shape[0]
    else:
        new_shape[0] = img_shape[0] - (img_shape[0] % multiple) + multiple
    if input_img.shape[1] % multiple == 0:
        new_shape[1] = input_img.shape[1]
    else:
        new_shape[1] = img_shape[1] - (img_shape[1] % multiple) + multiple

    zeros_mask = np.zeros(shape=new_shape, dtype=input_img.dtype)
    zeros_mask[:img_shape[0], :img_shape[1]] = input_img
    return zeros_mask


def preproc_packet(img):
    height, width, channels = img.shape
    half_width = width // 2

    l_img = img[:, :half_width]
    r_img = img[:, half_width:]
    l_img = adjustImage(l_img)
    r_img = adjustImage(r_img)
    return l_img, r_img


if __name__ == '__main__':
    weights_path = r"label_2024-01-23T02_58_31_weights.pth"
    folder_path = r"/home/zestiot/Desktop/Zestiot/PROJECTS/JSW/data/06_06"

    defect_path = r"C:\Users\kalyani chagala\Downloads\05_06 (1)\05_06_annotations"
    non_defect_path = r"C:\Users\kalyani chagala\workspace\python_scripts\JSW\real_belt_non_defects28_01_1"
    os.makedirs(defect_path, exist_ok=True)
    os.makedirs(non_defect_path, exist_ok=True)

    files = os.listdir(folder_path)

    device = torch.device('cpu')
    unet_model = Unet().to(device)
    unet_model.load_state_dict(
        torch.load(weights_path, map_location=device))
    unet_model.eval()

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0)
        )
    ])
    threshold = 0.85

    for file in files:
        img = cv2.imread(os.path.join(folder_path, file))
        org = img.copy()
        color_mask = img.copy()
        color_mask1 = img.copy()

        l_img, r_img = preproc_packet(img)

        l_tnsr, r_tnsr = data_transform(l_img).unsqueeze(0), data_transform(r_img).unsqueeze(0)
        with torch.no_grad():
            tnsr = torch.cat((l_tnsr, r_tnsr), dim=0).to(device)
            out_tnsr = unet_model(tnsr)
            l_out, r_out = out_tnsr.cpu().detach()
            l_out = l_out.view(l_out.shape[-2], l_out.shape[-1]).numpy()
            r_out = r_out.view(r_out.shape[-2], r_out.shape[-1]).numpy()

        out = np.hstack((l_out, r_out))

        threshold = out.max() * threshold
        temp_out = out * 255
        temp_out = temp_out.astype(np.uint8)

        _, thresh = cv2.threshold(
            temp_out, int(threshold * 255), 255, cv2.THRESH_BINARY)
        thresh_adjust = thresh[:img.shape[0], :img.shape[1]].copy()
        dilate_kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.dilate(thresh_adjust, dilate_kernel, iterations=4)

        contours, _ = cv2.findContours(
            img_dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        c = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        list_polygons = approx.tolist()
        filter_list = [co_ord[0] for co_ord in list_polygons]

        annotation_data = {
            "label": "object_class",
            "description": "",
            "points": filter_list,  # List of polygon vertices
            "group_id": None,
            "shape_type": "polygon",
            "mask": None
        }
        image_data = {
            "version": "5.4.1",
            "flags": {},
            "imagePath": "{}".format(file),
            "imageHeight": img.shape[0],
            "imageWidth": img.shape[1],
            "imageData": None,  # Optional, base64-encoded image data
            "shapes": [annotation_data]
        }

        json_data = json.dumps(image_data, indent=2)

        shutil.copyfile(os.path.join(folder_path, file), os.path.join(defect_path, file))
        with open(os.path.join(defect_path, "{}.json".format(file[:-4])), "w") as json_file:
            json_file.write(json_data)

    #cv2.imwrite(os.path.join(defect_path,file))
