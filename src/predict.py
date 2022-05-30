#!env python
import os
import glob
import time
import pickle
import argparse

from utils import *
from processing import *
from features import *
from model import *


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
                      default='gender_classifier.pkl')
  parser.add_argument('-x', '--exclude-feature',
                      help='Skips the feature extraction for the given feature.',
                      action='append', default=[])
  args = parser.parse_args()

  test_images = sorted(glob.glob(os.path.join(args.inputdir, '*.jpg')))

  selected_features = FEATURES.difference(args.exclude_feature)

  try:
    with open(args.classifier, 'rb') as clf_file:
      clf = pickle.load(clf_file)
  except Exception as e:
    print(f"Couldn't unpickle the classifier {args.classifier}\nError: {e}")

  results = []
  times = []

  for test_image in test_images:
    start_time = time.time()
    try:
      image = preprocess(test_image)
      bw_image = binarize(image)
      features = {}
      for feature in selected_features:
        features[feature] = run_feature_extraction(image, bw_image, feature)
      results.append(str(round(clf.predict(features))))
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
