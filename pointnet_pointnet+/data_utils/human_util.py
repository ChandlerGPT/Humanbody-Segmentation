
import numpy as np
import glob
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

DATA_PATH = os.path.join(ROOT_DIR,'data')
h_classes = [x.rstrip() for x in open(os.path.join(BASE_DIR, 'meta/human_class_names.txt'))]
h_class2label = {cls: i for i,cls in enumerate(h_classes)}
h_class2color = {1:	[255,0,127],
                 2:	[255,255,0],
                 3:	[0,255,255],
                 4: [255,0,255],
                 5: [0,127,127],
                 6: [127,127,255],
                 7: [0,0,255],
                 8: [255,0,0],
                 9: [127,255,127],
                 10:[127,127,127],
                 11:[0,127,255],
                 12:[255,127,0],
                 13:[127,255,255],
                 14:[0,255,0]} 
h_easy_view_labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
h_label2color = {h_classes.index(cls): h_class2color[cls] for cls in h_classes}
