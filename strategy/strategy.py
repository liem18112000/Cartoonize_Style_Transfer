import copy
import functools
import time
import PIL.Image
import numpy as np
import os
import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl

import cv2

class Strategy(object):
    
    def cartoonize(self, content, style):
        pass

class StyleTransferStrategy(Strategy):

    def cartoonize(self, content, style):

        (content_images, content_titles), (style_images, style_titles) = content, style

        content_layers = ['block5_conv2']

        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']

        # Load VGG Model & Create Extractor
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        extractor = StyleContentModel(vgg, style_layers, content_layers)

        # Style Transfering
        optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        style_weight = 0.001
        content_weight = 1000
        total_variation_weight = 30

        result_images = []
        for style_image in style_images:
            style_targets = extractor(style_image)['style']
            for content_image in content_images:
                image = tf.Variable(content_image)
                content_targets = extractor(content_image)['content']
                train_parameters = {
                    'optimizer': optimizer,
                    'loss_weight': (style_weight, content_weight, total_variation_weight),
                    'targets': (style_targets, content_targets),
                    'extractor': extractor,
                    'num_layer': (len(content_layers), len(style_layers))
                }
                image_style_tranfer(image, train_parameters, epochs=5)
                result_images.append(copy.deepcopy(image))

        return result_images

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, model, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(model, style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(
            inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

class Custom_Train(object):
    def __init__(self, image, optimizer, targets, loss_weight, extractor, num_layer):
        self._image = image
        self._optimizer = optimizer
        self._targets = targets
        self._loss_Weight = loss_weight
        self._extractor = extractor
        self._num_layer = num_layer

    def __call__(self):
        with tf.GradientTape() as tape:
            outputs = self._extractor(self._image)
            loss = style_content_loss(
                outputs,
                style_targets=self._targets[0],
                content_targets=self._targets[1],
                style_weight=self._loss_Weight[0],
                content_weight=self._loss_Weight[1], 
                num_content_layers=self._num_layer[0],
                num_style_layers=self._num_layer[1]
            ) + self._loss_Weight[2] * tf.image.total_variation(self._image)

        grad = tape.gradient(loss, self._image)
        self._optimizer.apply_gradients([(grad, self._image)])
        self._image.assign(clip_0_1(self._image))

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

"""## Model Utils"""

def vgg_layers(vgg, layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight, num_content_layers, num_style_layers):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                            for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                                for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

def image_style_tranfer(image, train_parameters, epochs=10, steps_per_epoch=100):
    start = time.time()
    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step = Custom_Train(
                image,
                train_parameters['optimizer'],
                train_parameters['targets'],
                train_parameters['loss_weight'],
                train_parameters['extractor'],
                train_parameters['num_layer']
            )
            train_step()
            print(".", end='')
        display.clear_output(wait=True)
        display.display(tensor_to_image(image))
        print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end-start))

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


class ImageProcessingStrategy(Strategy):

    def edge_detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        return edges

    def image_process(self, img, edge):
        color = cv2.bilateralFilter(img, 9, 250, 250)
        cartoon = cv2.bitwise_and(color, color, mask=edge)
        return cartoon

    def cartoonize(self, images, style = None):
        result_images = []
        for img in images[0]:
            edge = self.edge_detect(img)
            result_images.append(self.image_process(img, edge))
        return result_images

class Proxy(Strategy):
    def __init__(self, name = None):
        self._instance_name = name
        self._instance = None

    def setInstanceName(self, name):
        self._instance_name = str(name)

    def initRealInstance(self):
        print("Start lazy loading throung a proxy .....")
        if self._instance_name == 'Image_Processing':
            self._instance = ImageProcessingStrategy()
            return True
        elif self._instance_name == 'Style_Transfer':
            self._instance = StyleTransferStrategy()
            return True
        else:
            print(self._instance_name + " is not implemented in proxy")
            return False

    def cartoonize(self, images, style):
        if(self.initRealInstance()):
            return self._instance.cartoonize(images, style)
        else:
            print("Cartoonize fail")
            return []

    def getInstanceName(self):
        return self._instance_name
