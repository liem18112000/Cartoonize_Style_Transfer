from utils.image import ImageUtils as Loader
from strategy.strategy import Strategy

class CartoonizeInterface(object):

    def load_content_style_images(self, links, show = False):
        pass

    def cartoonize(self):
        pass

    def setStrategy(self, strategy):
        pass

    def setLoader(self, loader):
        pass

class Cartoonize(CartoonizeInterface):

    def __init__(self, strategy = None, loader = None):
        self._strategy = strategy
        self._loader = loader
        self._content = None
        self._style = None

    def _isLoaderAvailable(self):
        return self._loader is not None

    def _isStrategyAvailable(self):
        return self._strategy is not None

    def load_content_style_images(self, links, show = False):
        if not self._isLoaderAvailable():
            print("loader is not found")
            return

        self._content, self._style = self._loader.load_content_style_images(links[0], links[1], show)

    def cartoonize(self):
        if not self._isStrategyAvailable():
            print("strategy is not found")
            return

        if self._content is None or self._style is None:
            print("Content image and style image are not loaded")
            return 

        return self._strategy.cartoonize(self._content, self._style)

    def setStrategy(self, strategy):
        if isinstance(self._strategy, Strategy):
            self._strategy = strategy
        else:
            print("strategy is not compatible with Strategy::class")

    def setLoader(self, loader):
        if isinstance(self._loader, Loader):
            self._loader = loader
        else:
            print("loader is not compatible with Loader::class")

    def show_cartoonize_images(self):
        result_images = self.cartoonize()
        for index, result_image in enumerate(result_images):
            self._loader.imshow(result_image, 'Result Image ' + str(index + 1))
