import gradio as gr
import numpy as np

import torch

from types import SimpleNamespace
from PIL import Image

from asp.models.cpt_model import CPTModel
from asp.util.general_utils import parse_args
from asp.util.io_utils import download_weights

def preprocess_image(img):
    img_array = np.array(img)  # Shape: [H, W, C], dtype: uint8, values: [0, 255]

    if img_array.ndim == 2:  # Grayscale image
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA image
        img_array = img_array[:, :, :3]  # Discard the alpha channel

    img_array = np.transpose(img_array, (2, 0, 1))  # Shape: [C, H, W]
    img_array = img_array.astype(np.float32)  # Convert to float32
    img_array = img_array / 255.0 * 2.0 - 1.0  # Scale to [-1.0, 1.0]

    img_tensor = torch.from_numpy(img_array)  # Shape: [C, H, W]
    img_tensor = img_tensor.unsqueeze(0)  # Shape: [1, C, H, W]

    return img_tensor

def postprocess_tensor(tensor):
    output_img = tensor.squeeze(0).detach().cpu()
    output_img = output_img.clamp(-1.0, 1.0).float().numpy()
    output_img = (np.transpose(output_img, (1, 2, 0)) + 1) / 2.0 * 255.0
    output_img = output_img.astype(np.uint8)
    output_img = Image.fromarray(output_img)

    return output_img

def convert_he2ihc(output_stain, input_he_image_path):
    stain2folder_name = {
        "HER2 (Human Epidermal growth factor Receptor 2)": "ASP_pretrained/MIST_her2_lambda_linear",
        "ER (Estrogen Receptor)"                         : "ASP_pretrained/MIST_er_lambda_linear",
        "Ki67 (Antigen KI-67)"                           : "ASP_pretrained/MIST_ki67_lambda_linear",
        "PR (Progesterone Receptor)"                     : "ASP_pretrained/MIST_pr_lambda_linear",
    }

    input_img = Image.open(input_he_image_path).convert('RGB')
    original_img_size = input_img.size

    opt = SimpleNamespace(
        gpu_ids=None,
        isTrain=False,
        checkpoints_dir="../../checkpoints",
        name=stain2folder_name[output_stain],
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

    model.setup(opt)
    model.parallelize()
    model.eval()

    input_img = input_img.resize((1024, 1024))
    input_tensor = preprocess_image(input_img)

    model.set_input({
        "A": input_tensor, 
        "A_paths": input_he_image_path,
        "B": input_tensor,
        "B_paths": input_he_image_path,
    })
    model.test()
    visuals = model.get_current_visuals()

    output_img = postprocess_tensor(visuals['fake_B'])

    output_img = output_img.resize(original_img_size)
    print("np.shape(output_img)", np.shape(output_img))

    return output_img

def main():
    download_weights("1N_HOGU7FO4u-S1OD-bumZGyevYeucT4Q", "../../checkpoints/ASP_pretrained/MIST_her2_lambda_linear/latest_net_G.pth")
    download_weights("1j6xu8MAOVUaZuV4O5CqsBfMtH6-droys", "../../checkpoints/ASP_pretrained/MIST_er_lambda_linear/latest_net_G.pth")
    download_weights("10STHMS-GMkHMOJp_cJ44T66rRwKlUZyr", "../../checkpoints/ASP_pretrained/MIST_ki67_lambda_linear/latest_net_G.pth")
    download_weights("1APIrm3kqtPhhAIcU7pvfIcYpMjpsIlQ9", "../../checkpoints/ASP_pretrained/MIST_pr_lambda_linear/latest_net_G.pth")
    
    demo = gr.Interface(
        fn=convert_he2ihc,
        inputs=[
            gr.Dropdown(
                choices=["HER2 (Human Epidermal growth factor Receptor 2)", 
                 "ER (Estrogen Receptor)", 
                 "Ki67 (Antigen KI-67)", 
                 "PR (Progesterone Receptor)"], 
                label="Output Stain"
            ),
            gr.Image(type="filepath", label="Input H&E Image")
        ],
        outputs=gr.Image(label="Ouput IHC Image"),
        title="H&E-to-IHC Stain Translation",
        description="<h2>Stain your H&E (Hematoxylin and Eosin) images into IHC (ImmunoHistoChemistry) images automatically thanks to AI!</h2>",
        theme="ParityError/Interstellar"
    )

    demo.launch()

if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))

# python app.py