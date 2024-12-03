import os

import gradio as gr
import numpy as np

import random
import torch
from torchvision.transforms.functional import to_pil_image

from types import SimpleNamespace
from PIL import Image

from asp.models.cpt_model import CPTModel
from asp.data.base_dataset import get_transform
from asp.util.general_utils import parse_args
from asp.util.io_utils import download_weights

def transform_with_seed(input_img, transform, seed=123456):
    random.seed(seed)
    torch.manual_seed(seed)
    return transform(input_img)

def convert_he2ihc(input_he_image_path):
    input_img = Image.open(input_he_image_path).convert('RGB')
    original_img_size = input_img.size

    opt = SimpleNamespace(
        gpu_ids=None,
        isTrain=False,
        checkpoints_dir="../../checkpoints",
        # name="ASP_pretrained/BCI_her2_lambda_linear",
        name="ASP_pretrained/BCI_her2_zero_uniform",
        preprocess="scale_width_and_crop",
        nce_layers="0,4,8,12,16",
        nce_idt=False,
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG="resnet_6blocks",
        normG="instance",
        no_dropout=True,
        init_type="xavier",
        init_gain=0.02,
        no_antialias=False,
        no_antialias_up=False,
        weight_norm="spectral",
        netF="mlp_sample",
        netF_nc=256,
        no_flip=True,
        load_size=1024,
        crop_size=1024,
        direction="AtoB",
        flip_equivariance=False,
        epoch="latest",
        verbose=True
    )
    model = CPTModel(opt)

    transform = get_transform(opt)

    model.setup(opt)
    model.parallelize()
    model.eval()

    A = transform_with_seed(input_img, transform)
    model.set_input({
        "A": A.unsqueeze(0), 
        "A_paths": input_he_image_path,
        "B": A.unsqueeze(0),
        "B_paths": input_he_image_path,
    })
    model.test()
    visuals = model.get_current_visuals()

    output_img = to_pil_image(visuals['fake_B'].detach().cpu().squeeze(0))
    output_img = output_img.resize(original_img_size)
    print("np.shape(output_img)", np.shape(output_img))

    return output_img

def main():
    download_weights("1SMTeMprETgXAfJGXQz0LtgXXetfKXNaW", "../../checkpoints/ASP_pretrained/BCI_her2_zero_uniform/latest_net_G.pth")

    demo = gr.Interface(
        fn=convert_he2ihc,
        inputs=gr.Image(type="filepath"),
        outputs=gr.Image(),
        title="H&E to IHC, BIC HER2"
    )

    demo.launch()

if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))

# python app.py