import os
import cv2
import torch
import torch.nn as nn 
import numpy as np 
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import collections

weights_path = '/data2/wangjiajie/server_38/Zero-DCE/model200.pth' # checkpoints/model_176_0.pth
infer_img_path = '/data2/wangjiajie/server_38/Zero-DCE/DarkPair/Low/' # DarkPair/Low/
output_img_path = '/data2/wangjiajie/server_38/Zero-DCE/outputs0210/'
if not os.path.exists(output_img_path):
    os.makedirs(output_img_path)
image_resize_factor = None



class DCENet(nn.Module):
    """DCENet Module"""
    def __init__(self, n_filters=32):
        super(DCENet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=n_filters,
            kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            in_channels=n_filters, out_channels=n_filters,
            kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv3 = nn.Conv2d(
            in_channels=n_filters, out_channels=n_filters,
            kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv4 = nn.Conv2d(
            in_channels=n_filters, out_channels=n_filters,
            kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv5 = nn.Conv2d(
            in_channels=n_filters * 2, out_channels=n_filters,
            kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv6 = nn.Conv2d(
            in_channels=n_filters * 2, out_channels=n_filters,
            kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv7 = nn.Conv2d(
            in_channels=n_filters * 2, out_channels=24,
            kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        return enhance_image

# Create Model
model = DCENet()
model.cuda()
checkpoint = torch.load(weights_path)
d = collections.OrderedDict()
for key, value in checkpoint.items():
    tmp = key[7:]
    d[tmp] = value
model.load_state_dict(d)

image_files = glob(infer_img_path + '/*.png')
for image_file in image_files:
    with torch.no_grad():
        image_lowlight = Image.open(image_file)
        width, height = image_lowlight.size
        if image_resize_factor is not None:
            image_lowlight = image_lowlight.resize((width // image_resize_factor, height // image_resize_factor), Image.ANTIALIAS)
        lowlight = (np.asarray(image_lowlight) / 255.0)
        lowlight = torch.from_numpy(lowlight).float()
        lowlight = lowlight.permute(2, 0, 1)
        lowlight = lowlight.cuda().unsqueeze(0)
        enhanced = model(lowlight)
        enhanced = enhanced.squeeze().permute(1, 2, 0)
        image_enhance = enhanced.cpu().numpy()
        image_enhance *= 255

        image_ori = cv2.cvtColor(np.asarray(image_lowlight), cv2.COLOR_RGB2BGR)
        image_enh = cv2.cvtColor(np.asarray(image_enhance), cv2.COLOR_RGB2BGR)
        image_final = np.hstack([image_ori, image_enh])
        cv2.imwrite(image_file.replace(infer_img_path, output_img_path), image_final)
        