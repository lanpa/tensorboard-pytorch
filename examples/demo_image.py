import io
import urllib

import numpy as np
from PIL import Image, ImageDraw

from tensorboardX import SummaryWriter
from tensorboardX.proto.summary_pb2 import Summary

writer = SummaryWriter()

# Generated image - 'HWD' format

blue = Image.new('RGB', (240, 240), color='blue')
writer.add_image('blue', np.asarray(blue), dataformats='HWD')

# PNG file - 'HWC' format

screenshot = Image.open('../screenshots/scalar.png')
writer.add_image('screenshot', np.asarray(screenshot), dataformats='HWC')

# JPG format - 'HWD' format

url = ('https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/'
       'Cyanocitta_cristata_blue_jay.jpg/792px-Cyanocitta_cristata_blue_jay.jpg')
jay = Image.open(urllib.urlopen(url))
writer.add_image('jay', np.asarray(jay), dataformats='HWD')

# Avoid round-trip (image -> array -> image) by adding summary directly

def summary_image(img):
    output = io.BytesIO()
    img.save(output, format='PNG')
    encoded = output.getvalue()
    output.close()
    return Summary.Image(
        height=img.height,
        width=img.width,
        colorspace=len(img.getbands()),
        encoded_image_string=encoded)

def add_image(writer, tag, img):
    summary = Summary(value=[Summary.Value(tag=tag, image=summary_image(img))])
    writer.file_writer.add_summary(summary)

# Detection bounding box

jay_detect = jay.copy()
ImageDraw.Draw(jay_detect).rectangle(((505, 154), (550, 204)), outline="white")
add_image(writer, "jay-detect", jay_detect)

# Flip

screenshot_flipped = screenshot.transpose(Image.FLIP_LEFT_RIGHT)
add_image(writer, "screenshot-flipped", screenshot_flipped)

writer.close()
