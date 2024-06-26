import os
import os.path
import numpy as np
import xml.etree.ElementTree as xmlET
from PIL import Image, ImageDraw

#'1': 'people', '2': 'people','3': 'bicycle', '4': 'car', '5': 'car',
# 6':'others','7':'others','8':'others','9':'others','10': 'motor','11':'others'

classes = ('__background__', # always index 0
           'ignored regions','pedestrian', 'people','bicycle','car','van','truck','tricycle','awning-tricycle',
           'bus','motor', 'others')

# 把下面的路径改为自己的路径即可
file_path_img = r'E:\dataset\VisDrone2019\VisDrone2019-DET-test-dev\images'
file_path_xml = r'E:\dataset\VisDrone2019\VisDrone2019-DET-test-dev\xml'
save_file_path = r'E:\dataset\VisDrone2019\VisDrone2019-DET-test-dev\image_ann_show'

pathDir = os.listdir(file_path_xml)
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    tree = xmlET.parse(os.path.join(file_path_xml, filename))
    objs = tree.findall('object')
    num_objs = len(objs)
    boxes = np.zeros((num_objs, 5), dtype=np.uint16)

    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)

        cla = obj.find('name').text
        label = classes.index(cla)

        boxes[ix, 0:4] = [x1, y1, x2, y2]
        boxes[ix, 4] = label

    image_name = os.path.splitext(filename)[0]
    img = Image.open(os.path.join(file_path_img, image_name + '.jpg'))

    draw = ImageDraw.Draw(img)
    for ix in range(len(boxes)):
        xmin = int(boxes[ix, 0])
        ymin = int(boxes[ix, 1])
        xmax = int(boxes[ix, 2])
        ymax = int(boxes[ix, 3])
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))
        draw.text([xmin, ymin], classes[boxes[ix, 4]], (255, 0, 0))

    img.save(os.path.join(save_file_path, image_name + '.png'))