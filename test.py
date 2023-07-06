import argparse
import cv2
import numpy as np
import os
import torch
from models.DITN_Real import DITN_Real


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='DITN_Real')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 4')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--indir', default='')
    parser.add_argument('--outdir', default='')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DITN_Real(upscale=args.scale)
    pretrained_model = torch.load(args.model_path, map_location='cpu')
    if 'params' in pretrained_model.keys():
        model.load_state_dict(pretrained_model['params'], strict=True)
    else:
        model.load_state_dict(pretrained_model, strict=True)

    model.eval()
    model = model.to(device)
    os.makedirs(args.outdir, exist_ok=True)

    for image_lq_name in os.listdir(args.indir):
        image_lq_path = os.path.join(args.indir, image_lq_name)

        img_lq = cv2.imread(image_lq_path, cv2.IMREAD_COLOR).astype(
            np.float32) / 255.

        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]],
                                (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            output = model(img_lq)

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(os.path.join(args.outdir, os.path.splitext(image_lq_name)[0]+'_{}.png'.format(args.name)), output)

if __name__ == '__main__':
    main()