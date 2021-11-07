import argparse
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True, help='Image Directory for SVHN dataset')
args = parser.parse_args()

# reference: https://github.com/k-chuang/tf-svhn
class DigitStructWrapper:
    """
    Wrapper for the H5PY digitStruct files from the SVHN dataset
    Creates an array of dictionaries containing the filename and bounding boxes for every digit in the image.
    Adapted from https://github.com/hangyao
    """

    def __init__(self, inf):
        self.inf = h5py.File(inf, 'r')
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']

    def get_name(self, n):
        """Return the name of the n(th) digit struct"""
        if h5py.__version__ <= '2.9.0':
          return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]].value])
        else:
          return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]]])

    def get_attribute(self, attr):
        """Helper function for dealing with one vs. multiple bounding boxes"""
        if h5py.__version__ <= '2.9.0':
          if (len(attr) > 1):
              attr = [self.inf[attr.value[j].item()].value[0][0] for j in range(len(attr))]
          else:
              attr = [attr.value[0][0]]
        else:
          if (len(attr) > 1):
              attr = [self.inf[attr[j].item()][0][0] for j in range(len(attr))]
          else:
              attr = [attr[0][0]]
        return attr

    def get_bbox(self, n):
        """Return a dict containing the data from the n(th) bbox"""
        bbox = {}
        bb = self.digitStructBbox[n].item()
        bbox['height'] = self.get_attribute(self.inf[bb]["height"])
        bbox['label'] = self.get_attribute(self.inf[bb]["label"])
        bbox['left'] = self.get_attribute(self.inf[bb]["left"])
        bbox['top'] = self.get_attribute(self.inf[bb]["top"])
        bbox['width'] = self.get_attribute(self.inf[bb]["width"])
        return bbox

    def get_item(self, n):
        """Return the name and bounding boxes of a single image"""
        s = self.get_bbox(n)
        s['name'] = self.get_name(n)
        return s

    def unpack(self):
        """Returns a list of dicts containing all the bounding boxes"""
        return [self.get_item(i) for i in range(len(self.digitStructName))]

    def unpack_all(self):
        pictDat = self.unpack()
        result = []
        structCnt = 1
        for i in tqdm(range(len(pictDat))):
            item = {'filename': pictDat[i]["name"]}
            figures = []
            for j in range(len(pictDat[i]['height'])):
                figure = {}
                figure['height'] = pictDat[i]['height'][j]
                figure['label'] = pictDat[i]['label'][j]
                figure['left'] = pictDat[i]['left'][j]
                figure['top'] = pictDat[i]['top'][j]
                figure['width'] = pictDat[i]['width'][j]
                figures.append(figure)
            structCnt = structCnt + 1
            item['boxes'] = figures
            result.append(item)
        return result

rootdir=args.dir
d=DigitStructWrapper(f'{rootdir}/digitStruct.mat')
f=open(f'{rootdir}/annotation.txt','w')
f2=open(f'{rootdir}/annotation_box.txt','w')
for i in tqdm(range(len(d.digitStructName))):
    single_data=d.get_item(i)
    # Image Shape
    img_shape=Image.open(os.path.join(rootdir,single_data['name'])).size
    w,h=img_shape
    half_w=w/2
    half_h=h/2
    # Digit Label
    filename = single_data["name"]
    label = [str(int(i)) if i!=10 else "0" for i in single_data["label"]]
    label = ''.join(label)
    line=f'{single_data["name"]} {label}\n'
    f.writelines([line])
    # Digit Bounding Box
    xmin=(min([float(item) for item in single_data['left']])-half_w)/half_w
    ymin=(min([float(item) for item in single_data['top']])-half_h)/half_h
    xmax=(max([float(a)+float(b) for a,b in zip(single_data['left'], single_data['width'])])-half_w)/half_w
    ymax=(max([float(a)+float(b) for a,b in zip(single_data['top'], single_data['height'])])-half_h)/half_h
    line2=f'{single_data["name"]} {label} {xmin} {ymin} {xmax} {ymax}\n' 
    f2.writelines([line2])
f.close()
f2.close()