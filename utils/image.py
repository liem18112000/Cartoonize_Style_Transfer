import functools
import time
import PIL.Image
import numpy as np
import os
import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (24, 24)
mpl.rcParams['axes.grid'] = False





class ImageUtils(object):

    def load_img(self, path_to_img, max_dim=512):
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img


    def imshow(self, image, title=None):
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)

        plt.imshow(image)
        if title:
            plt.title(title)

    def load_content_style_images(self, content_links, style_links, show_images = False):
        
        content_images, content_titles  = self.load_images_from_link(content_links)
        style_images, style_titles      = self.load_images_from_link(style_links)

        if show_images:
            print("Content Images : ")
            self.show_all_images(content_images, content_titles)

            print("Style Images : ")
            self.show_all_images(style_images, style_titles)

        return (content_images, content_titles), (style_images, style_titles) 

    def load_images_from_link(self, image_links):
        images = []
        for index, link in enumerate(image_links):
            titles = []
            path = tf.keras.utils.get_file('girl' + '_' + str(index) + '.jpg', link)
            titles.append('girl' + '_' + str(index) + '.jpg')
            images.append(self.load_img(path))

        return images, titles


    def show_all_images(self, images, titles = []):
        plot_len = len(images)
        for i in range(plot_len):
            plt.subplot(1, plot_len, i + 1)
            self.imshow(
                images[i]
                # titles[i]
            )

