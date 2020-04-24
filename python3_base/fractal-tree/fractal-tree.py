# Adapted from http://rosettacode.org/wiki/Fractal_tree#Python
#   to parameterise, and add colour.
#   http://pillow.readthedocs.org/
#   Author: Alan Richmond, Python3.codes, and others (Rosettacode)

import math, colorsys
from PIL import Image, ImageDraw

spread = 17                 # how much branches spread apart
width, height = 1000, 800   # window size
maxd = 12                   # maximum recursion depth
len = 9.0                   # branch length factor

#   http://pillow.readthedocs.org/en/latest/reference/Image.html
img = Image.new('RGB', (width, height))
#   http://pillow.readthedocs.org/en/latest/reference/ImageDraw.html
d = ImageDraw.Draw(img)

# This function calls itself to add sub-trees
def drawTree(x1, y1, angle, depth):
    if depth > 0:
        #       compute this branch's next endpoint
        x2 = x1 + int(math.cos(math.radians(angle)) * depth * len)
        y2 = y1 + int(math.sin(math.radians(angle)) * depth * len)

        #   https://docs.python.org/2/library/colorsys.html
        (r, g, b) = colorsys.hsv_to_rgb(float(depth) / maxd, 1.0, 1.0)
        R, G, B = int(255 * r), int(255 * g), int(255 * b)

        #       draw the branch
        d.line([x1, y1, x2, y2], (R, G, B), depth)

        #       and append 2 trees by recursion
        drawTree(x2, y2, angle - spread, depth - 1)
        drawTree(x2, y2, angle + spread, depth - 1)

#   Start drawing!
drawTree(width / 2, height * 0.9, -90, maxd)
img.show()
img.save(".png", "PNG")