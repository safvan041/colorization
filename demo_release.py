import argparse
import os
import logging
import matplotlib.pyplot as plt
import torch

from colorizers import *  # Assuming colorizers.py contains the necessary colorization models and utilities

def load_colorizers(use_gpu):
    """
    Load the colorization models and move them to GPU if specified.
    """
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    if use_gpu:
        colorizer_eccv16.cuda()
        colorizer_siggraph17.cuda()
    return colorizer_eccv16, colorizer_siggraph17

def process_image(img_path, size, use_gpu):
    """
    Load and preprocess the image.
    """
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    img = load_img(img_path)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(size, size))
    if use_gpu:
        tens_l_rs = tens_l_rs.cuda()
    return img, tens_l_orig, tens_l_rs

def colorize_images(colorizer_eccv16, colorizer_siggraph17, tens_l_rs, tens_l_orig):
    """
    Apply colorization models and return the colorized images.
    """
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig, 0*tens_l_orig), dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
    return img_bw, out_img_eccv16, out_img_siggraph17

def save_and_display_results(img, img_bw, out_img_eccv16, out_img_siggraph17, save_prefix, output_format):
    """
    Save the colorized images and display results.
    """
    plt.imsave(f'{save_prefix}_eccv16.{output_format}', out_img_eccv16)
    plt.imsave(f'{save_prefix}_siggraph17.{output_format}', out_img_siggraph17)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(img_bw)
    plt.title('Input')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(out_img_eccv16)
    plt.title('Output (ECCV 16)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(out_img_siggraph17)
    plt.title('Output (SIGGRAPH 17)')
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_path', type=str, default='imgs/test.png', help='Path to the input image')
    parser.add_argument('--use_gpu', action='store_true', help='Whether to use GPU')
    parser.add_argument('-o', '--save_prefix', type=str, default='saved', help='Prefix for saved output files')
    parser.add_argument('--img_size', type=int, default=256, help='Size to which the image will be resized before processing')
    parser.add_argument('--output_format', type=str, choices=['png', 'jpg'], default='png', help='Format for saving the output images')
    opt = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        colorizer_eccv16, colorizer_siggraph17 = load_colorizers(opt.use_gpu)
        img, tens_l_orig, tens_l_rs = process_image(opt.img_path, opt.img_size, opt.use_gpu)
        img_bw, out_img_eccv16, out_img_siggraph17 = colorize_images(colorizer_eccv16, colorizer_siggraph17, tens_l_rs, tens_l_orig)
        save_and_display_results(img, img_bw, out_img_eccv16, out_img_siggraph17, opt.save_prefix, opt.output_format)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        exit(1)

if __name__ == "__main__":
    main()
