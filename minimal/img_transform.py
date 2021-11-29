"""
    This file contains the implementation of the DeepDream algorithm.

    If you have problems understanding any parts of the code,
    go ahead and experiment with functions in the playground.py file.
"""

import os
import argparse
import shutil
import time

from pdb import set_trace as pdb

import numpy as np
import torch
import cv2 as cv
from torchvision import models

import matplotlib.pyplot as plt


import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "pytorch-grad-cam/"))
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import utils.utils as utils
from utils.constants import *
import utils.video_utils as video_utils

targidx=296

# loss.backward(layer) <- original implementation did it like this it's equivalent to MSE(reduction='sum')/2
def gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration,mask):

    outputs= []
    def hook(module, input, output):
        outputs.append(output)
    model.features[-11].register_forward_hook(hook)

    # Step 0: Feed forward pass
    #out = model(input_tensor)[0]#.softmax(0)

    # Step 0: Feed forward pass

    """
    grayscale_cam = cam(input_tensor=input_tensor, target_category=258)
    grayscale_cam = grayscale_cam[0, :]
    img = utils.pytorch_output_adapter(input_tensor)
    img= utils.post_process_numpy_img(img)
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    plt.imshow(visualization);plt.show()
    """

    """
    targ=torch.zeros_like(out)
    targ[259]=1
    """

    # Step 1: Grab activations/feature maps of interest
    out = model(input_tensor)[0]
    #activations = [out[layer_id_to_use] for layer_id_to_use in layer_ids_to_use]
    activations = outputs

    # Step 2: Calculate loss over activations
    mask=(torch.tensor(mask)).float()
    losses = []
    for layer_activation in activations:
        # Use torch.norm(torch.flatten(layer_activation), p) with p=2 for L2 loss and p=1 for L1 loss.
        # But I'll use the MSE as it works really good, I didn't notice any serious change when going to L1/L2.
        # using torch.zeros_like as if we wanted to make activations as small as possible but we'll do gradient ascent
        # and that will cause it to actually amplify whatever the network "sees" thus yielding the famous DeepDream look
        loss_component = torch.nn.MSELoss(reduction='mean')(layer_activation, torch.zeros_like(layer_activation))
        losses.append(loss_component)
    loss = torch.mean(torch.stack(losses))*1
    tmp= out.softmax(0)[targidx]#*1e2
    loss+=tmp*(2e0 if tmp>.01 else 1e4)
    print(out.softmax(0)[targidx])
    print(out[targidx])
    loss.backward()
    """
    out=out.softmax(0)
    targ=torch.zeros_like(out)
    targ[targidx]=1
    loss = (out-targ).abs().mean()
    print(out[targidx])
    loss.backward()
    """

    # Step 3: Process image gradients (smoothing + normalization)
    grad = input_tensor.grad.data

    # Applies 3 Gaussian kernels and thus "blurs" or smoothens the gradients and gives visually more pleasing results
    # sigma is calculated using an arbitrary heuristic feel free to experiment
    sigma = ((iteration + 1) / config['num_gradient_ascent_iterations']) * 2.0 + config['smoothing_coefficient']
    smooth_grad = utils.CascadeGaussianSmoothing(kernel_size=9, sigma=sigma)(grad)  # "magic number" 9 just works well

    # Normalize the gradients (make them have mean = 0 and std = 1)
    # I didn't notice any big difference normalizing the mean as well - feel free to experiment
    g_std = torch.std(smooth_grad)
    g_mean = torch.mean(smooth_grad)
    smooth_grad = smooth_grad - g_mean
    smooth_grad = smooth_grad / g_std

    # Step 4: Update image using the calculated gradients (gradient ascent step)
    input_tensor.data += mask[None,None,:,:,0]*(config['lr'] * smooth_grad)
    #parser.add_argument("--lr", type=float, help="Learning rate i.e. step size in gradient ascent", default=0.09)

    # Step 5: Clear gradients and clamp the data (otherwise values would explode to +- "infinity")
    input_tensor.grad.data.zero_()
    input_tensor.data = torch.max(torch.min(input_tensor, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)

def deep_dream_static_image(config, img):
    #model = utils.fetch_and_prepare_model(config['model_name'], config['pretrained_weights'], DEVICE)
    model = models.vgg19(pretrained=True)
    layer_ids_to_use = None
    """
    try:
        layer_ids_to_use = [model.layer_names.index(layer_name) for layer_name in config['layers_to_use']]
    except Exception as e:  # making sure you set the correct layer name for this specific model
        print(f'Invalid layer names {[layer_name for layer_name in config["layers_to_use"]]}.')
        print(f'Available layers for model {config["model_name"]} are {model.layer_names}.')
        return
    """

    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    #get model working in same way for full vgg net and add additional loss of maximizing targ class score

    if img is None:  # load either the provided image or start from a pure noise image
        img_path = utils.parse_input_file(config['input'])
        # load a numpy, [0, 1] range, channel-last, RGB image
        img = utils.load_image(img_path, target_shape=config['img_width'])

    mask=(plt.imread("mask.png")[:,:,-1]>0).astype(float)[:,:,None]
    #mask=(plt.imread("headmask.png")[:,:,-1]>0).astype(float)[:,:,None]

    #plt.imshow(mask*img);plt.show()
    img = utils.pre_process_numpy_img(img)
    base_shape = img.shape[:-1]  # save initial height and width

    # Note: simply rescaling the whole result (and not only details, see original implementation) gave me better results
    # Going from smaller to bigger resolution (from pyramid top to bottom)
    for pyramid_level in range(config['pyramid_size']):
        new_shape = utils.get_new_shape(config, base_shape, pyramid_level)
        img = cv.resize(img, (new_shape[1], new_shape[0]))
        input_tensor = utils.pytorch_input_adapter(img, DEVICE)

        for iteration in range(config['num_gradient_ascent_iterations']):
            print(iteration)
            gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration,mask)

            img = utils.pytorch_output_adapter(input_tensor)
            img= utils.post_process_numpy_img(img)
            #utils.save_and_maybe_display_image(config, img)

        grayscale_cam = cam(input_tensor=input_tensor, target_category=targidx)
        grayscale_cam = grayscale_cam[0, :]
        img = utils.pytorch_output_adapter(input_tensor)
        img= utils.post_process_numpy_img(img)
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        plt.imsave("vis.png",visualization)

        img = utils.pytorch_output_adapter(input_tensor)

    return utils.post_process_numpy_img(img),visualization

def run(img1="dog.jpg",targ_class=296,lr=.09,display=False,num_iter=10):
    # Only a small subset is exposed by design to avoid cluttering
    parser = argparse.ArgumentParser()

    # Common params
    parser.add_argument("--input", type=str, help="Input IMAGE or VIDEO name that will be used for dreaming", default='dog.jpg')
    parser.add_argument("--img_width", type=int, help="Resize input image to this width", default=256)
    parser.add_argument("--layers_to_use", type=str, nargs='+', help="Layer whose activations we should maximize while dreaming", default=['relu4_3'])
    parser.add_argument("--model_name", choices=[m.name for m in SupportedModels],
                        help="Neural network (model) to use for dreaming", default=SupportedModels.VGG16_EXPERIMENTAL.name)
    parser.add_argument("--pretrained_weights", choices=[pw.name for pw in SupportedPretrainedWeights],
                        help="Pretrained weights to use for the above model", default=SupportedPretrainedWeights.IMAGENET.name)

    # Main params for experimentation (especially pyramid_size and pyramid_ratio)
    parser.add_argument("--pyramid_size", type=int, help="Number of images in an image pyramid", default=1)
    parser.add_argument("--pyramid_ratio", type=float, help="Ratio of image sizes in the pyramid", default=1.8)
    parser.add_argument("--num_gradient_ascent_iterations", type=int, help="Number of gradient ascent iterations", default=num_iter)
    parser.add_argument("--lr", type=float, help="Learning rate i.e. step size in gradient ascent", default=lr)

    # You usually won't need to change these as often
    parser.add_argument("--spatial_shift_size", type=int, help='Number of pixels to randomly shift image before grad ascent', default=32)
    parser.add_argument("--smoothing_coefficient", type=float, help='Directly controls standard deviation for gradient smoothing', default=0.5)
    parser.add_argument("--use_noise", action='store_true', help="Use noise as a starting point instead of input image (default False)")
    args = parser.parse_args()

    # Wrapping configuration into a dictionary
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    config['dump_dir'] = OUT_IMAGES_PATH
    config['dump_dir'] = os.path.join(config['dump_dir'], f'{config["model_name"]}_{config["pretrained_weights"]}')
    config['input_name'] = os.path.basename(config['input'])
    config['should_display'] = display

    print('Dreaming started!')
    img,vis = deep_dream_static_image(config, img=None)  # img=None -> will be loaded inside of deep_dream_static_image
    dump_path,dump_img = utils.save_and_maybe_display_image(config, img)
    print(f'Saved DeepDream static image to: {os.path.relpath(dump_path)}\n')
    return dump_img,vis
