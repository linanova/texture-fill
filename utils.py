import os.path
from PIL import Image

def parse_image_argument(parser, arg):
    """ Parser helper for image file arguments. """
    if not os.path.isfile(arg):
        parser.error("file %s does not exist" % arg)
    else:
        try:
            img = Image.open(arg).convert('RGB')
            return img
        except IOError:
            parser.error("could not open file %s" % arg)