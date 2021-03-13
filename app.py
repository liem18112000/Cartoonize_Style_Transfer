from strategy.strategy import StyleTransferStrategy as Strategy

from cartoonize.cartoonize import Cartoonize

from utils.image import ImageUtils as Loader

import argparse

def parser():
    
    parser = argparse.ArgumentParser(description='Cartoonize image with style image and content image.')
    parser.add_argument('content', help='path to content file images')
    parser.add_argument('style', help='path to style file images')
    parser.add_argument('save_dir', help='path to save result images')
    args = parser.parse_args()
    return args
    

def file_analyze(filename):

    inputs = []
    f = open(filename, 'r')
    for line in f:
        inputs.append(line.rstrip("\n"))

    return inputs

def app():
    """## Content image and Style application"""
    args = parser()
    content_links, style_links = file_analyze(args.content), file_analyze(args.style)

    tool = Cartoonize(
        strategy    = Strategy(),
        loader      = Loader(),
    )

    tool.load_content_style_images(
        links       = (content_links, style_links), 
    )

    tool.show_cartoonize_images(save_dir = args.save_dir)

test()
