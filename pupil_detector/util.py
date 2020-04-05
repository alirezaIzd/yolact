import PIL
import numpy as np
import numpy as np

from PIL import Image, ImageDraw

try:
    import Image
except ImportError:
    from PIL import Image


def shape_to_mask(img_shape, points, shape_type=None,

                  line_width=10, point_size=5):

    mask = np.zeros(img_shape[:2], dtype=np.uint8)

    mask = PIL.Image.fromarray(mask)

    draw = PIL.ImageDraw.Draw(mask)

    xy = [tuple(point) for point in points]

    if shape_type == 'circle':

        assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'

        (cx, cy), (px, py) = xy

        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)

        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)

    elif shape_type == 'rectangle':

        assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'

        draw.rectangle(xy, outline=1, fill=1)

    elif shape_type == 'line':

        assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'

        draw.line(xy=xy, fill=1, width=line_width)

    elif shape_type == 'linestrip':

        draw.line(xy=xy, fill=1, width=line_width)

    elif shape_type == 'point':

        assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'

        cx, cy = xy[0]

        r = point_size

        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)

    else:

        assert len(xy) > 2, 'Polygon must have points more than 2'

        draw.polygon(xy=xy, outline=1, fill=1)

    mask = np.array(mask, dtype=bool)

    return mask
