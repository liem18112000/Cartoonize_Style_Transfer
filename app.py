from strategy.strategy import StyleTransferStrategy , ImageProcessingStrategy, Proxy
from cartoonize.cartoonize import Cartoonize
from utils.image import ImageUtils, ImageProcessingUtils
import argparse
from utils.singleton import Singleton 
import os

@Singleton
class Application(object):

    def __init__(self):
        self._objects = {}

    def createParser(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        parser = argparse.ArgumentParser(description='Cartoonize image with style image and content image.')
        parser.add_argument('--strategy', help='choose a strategy to cartoonize', default="Image_Processing",)
        parser.add_argument('--content', help='path to content file images', default=dir_path + "/input/content.txt")
        parser.add_argument('--style', help='path to style file images', default=dir_path + "/input/style.txt")
        parser.add_argument('--save_dir', help='path to save result images', default=dir_path + "/output")
        return parser

    def getArgs(self, parser):
        return vars(parser.parse_args())
        
    def file_analyze(self, filename):
        inputs = []
        f = open(filename, 'r')
        for line in f:
            inputs.append(line.rstrip("\n"))

        return inputs

    def argsDispatch(self):
        args = self.getArgs(self.createParser())

        self._objects['strategy'] = Proxy(args['strategy'])
        self._objects['content'] = self.file_analyze(args['content'])
        self._objects['style'] = self.file_analyze(args['style'])
        self._objects['save_dir'] = args['save_dir']

        if self._objects['strategy'].getInstanceName() == "Image_Processing":
            self._objects['loader'] = ImageProcessingUtils()
        elif self._objects['strategy'].getInstanceName() == "Style_Transfer":
            self._objects['loader'] = ImageUtils()
        else:
            print("Strategy doesn't support Loader utils")
            return False

        return True

    def run(self):
        """## Content image and Style application"""
        
        if self.argsDispatch():

            tool = Cartoonize(
                strategy        = self._objects['strategy'],
                loader          = self._objects['loader'],
            )

            tool.load_content_style_images(
                links           = (self._objects['content'], self._objects['style'])
            )

            tool.show_cartoonize_images(
                save_dir        = self._objects['save_dir'],
            )

        else:
            print("Cartoonize fail")
