#!env python
import os
import glob
import time
import pickle
import argparse

from utils import *
from processing import *
from features import *


def main():
  parser = argparse.ArgumentParser(description='A male/female handwriting classifier.')
  parser.add_argument('-i', '--inputdir',
                      help='The path to the input directory. Which images are read from.',
                      default='test')
  parser.add_argument('-o', '--outputdir',
                      help='The path to the output directory. Where results and times will be reported.',
                      default='out')
  parser.add_argument('-c', '--classifier',
                      help='The path to the pickled classifier to use.',
                      default='hinge_clf.pkl')
  args = parser.parse_args()

  test_images = sorted(glob.glob(os.path.join(args.inputdir, '*.jpg')))

  try:
    with open(args.classifier, 'rb') as clf_file:
      clf = pickle.load(clf_file)
  except Exception as e:
    print(f"Couldn't unpickle the classifier {args.classifier}\nError: {e}")

  results = []
  times = []

  for test_image in test_images:
    image = imread(test_image)
    # cv2.imshow('f', image)
    # cv2.waitKey(0)
    start_time = time.time()
    try:
      assert image is not None, f"{test_image} couldn't be read."
      imwrite('f.jpg', preprocess(image))
      features = hinge(imread('f.jpg'))
      prediction = clf.predict([features])[0]
      print(prediction)
      results.append(str(round(prediction)))
    except Exception as e:
      print(e)
      results.append('-1')
    times.append(f'{time.time() - start_time:.2f}')

  with open(os.path.join(args.outputdir, 'results.txt'), 'w') as results_file:
    results_file.write('\n'.join(results))

  with open(os.path.join(args.outputdir, 'times.txt'), 'w') as times_file:
    times_file.write('\n'.join(times))

if __name__ == '__main__':
  main()
