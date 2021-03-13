from strategy.strategy import StyleTransferStrategy as Strategy

from cartoonize.cartoonize import Cartoonize

from utils.image import ImageUtils as Loader

def app():
    """## Content image and Style application"""

    content_links = [
        "https://scontent.xx.fbcdn.net/v/t1.15752-0/p180x540/158437160_240379114477751_4224753091836823568_n.jpg?_nc_cat=101&ccb=1-3&_nc_sid=f79d6e&_nc_ohc=GKh4DsQjXz8AX-gIzzr&_nc_ad=z-m&_nc_cid=0&_nc_ht=scontent.xx&tp=6&oh=f9aa960e5e4e3e4c2eddc02dfbb7e0d1&oe=606E060B",
        # "https://cdn2.stylecraze.com/wp-content/uploads/2013/06/nanako-matsushima.jpg",
        # "https://i.pinimg.com/564x/fc/48/af/fc48af3dc61155d0382f5d095694c585.jpg",
        # "https://i.pinimg.com/originals/bf/17/05/bf170507466915c157bed4fbd6c59a78.jpg",
        # "https://nypost.com/wp-content/uploads/sites/2/2020/12/yael-most-beautiful-video.jpg",
        # "https://s.yimg.com/ny/api/res/1.2/6fh8dX7HxCqWj0fCxMIKfQ--/YXBwaWQ9aGlnaGxhbmRlcjtoPTY2Ng--/https://s.yimg.com/cd/resizer/2.0/original/-Q7ql8v_Hy83ubHz_N1KOxjFLbo",
        # "https://i.pinimg.com/564x/57/14/96/571496d0e562669c7e3b39373cc3b4af.jpg"
    ]

    style_links = [
        "https://blogphotoshop.com/wp-content/uploads/2019/03/nhung-hinh-anh-anime-nu-dep-nhat-2.jpg",
        "https://i.pinimg.com/originals/79/b5/da/79b5da63f1c2cb9ce94e3801621ebb60.png",
        "https://i.pinimg.com/736x/2c/cc/25/2ccc2516f0e795a1f504ce54872c4f73.jpg",
        "https://media.overstockart.com/optimized/cache/data/product_images/VG485-1000x1000.jpg",
        "https://i.pinimg.com/originals/88/8a/a9/888aa921251ebd4de7a4833b715dee33.jpg",
        "https://i.pinimg.com/originals/8e/f8/0c/8ef80cdca4b6469c34e2645177c65929.jpg"
    ]

    tool = Cartoonize(
        strategy    = Strategy(),
        loader      = Loader(),
    )

    tool.load_content_style_images(
        links       = (content_links, style_links), 
    )

    tool.show_cartoonize_images()

app()
