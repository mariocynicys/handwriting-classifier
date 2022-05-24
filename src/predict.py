#!env python
import os
import glob
import argparse

from processing import *
from features import *
from helpers import *
from utils import *


def main():
  parser = argparse.ArgumentParser(description='A male/female handwriting classifier.')
  parser.add_argument('-i', '--inputdir',
                      help='The path to the input directory. Which images are read from.',
                      default='test')
  parser.add_argument('-o', '--outputdir',
                      help='The path to the output directory. Where results and times will be reported.',
                      default='out')
  args = parser.parse_args()

  test_images = sorted(glob.glob(os.path.join(args.inputdir, '*.jpg')))

  for test_image in test_images:
    image = preprocess(test_image)
    bw_image = binarize(image)
    for feature in FEATURES:
      pass

FEATURES = [
  'lbp',
  'hog',
  'glcm',
  'cold',
  'hinge',
  'slopes_and_curves',
  'chain_codes_and_pairs',
]

if __name__ == '__main__':
  main()
