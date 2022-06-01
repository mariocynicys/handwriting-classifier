#!env python
import os
import glob
import time
import pickle
import argparse
import warnings

from utils import *
from processing import *
from features import *
from model import *

class CustomArgParseFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter):
    """A custom formatter that combines the features of multiple base classes.
    This gives us defaults for each argument in the help text, plus it preserves
    whitespace in the description field.
    Credits: https://github.com/shaka-project/shaka-streamer/blob/4d1341df12309d179d067dc9bf634dc3f7a7c865/shaka-streamer#L36-L44
    """
    pass


def main():
  parser = argparse.ArgumentParser(description='A He/She handwriting classifier.',
                                   formatter_class=CustomArgParseFormatter)
  parser.add_argument('-i', '--inputdir',
                      help='The path to the input directory. Which images are read from.',
                      default='test')
  parser.add_argument('-o', '--outputdir',
                      help='The path to the output directory. Where results and times will be reported.',
                      default='out')
  parser.add_argument('-c', '--classifier',
                      help='The path to the pickled classifier to use.',
                      default='gender_classifier.pkl')
  parser.add_argument('-v', '--verbose',
                      help='Tell the classification of each image in realtime besides writing'
                      ' to results.txt and times.txt.',
                      action='store_true')
  parser.add_argument('-x', '--exclude-feature',
                      help='Skips the feature extraction for the given feature.',
                      action='append', default=[])
  parser.add_argument('-u', '--only-use',
                      help='Only use the passed features for classification.'
                      ' Overrides the exclude-features option.',
                      action='append', default=[])
  args = parser.parse_args()

  test_images = sorted(glob.glob(os.path.join(args.inputdir, '*.jpg')))

  selected_features = FEATURES.difference(args.exclude_feature)
  if args.only_use:
    selected_features = FEATURES.intersection(args.only_use)

  try:
    with open(args.classifier, 'rb') as clf_file:
      clf = pickle.load(clf_file)
  except Exception as e:
    print(f"Couldn't unpickle the classifier {args.classifier}\nError: {e}")

  results = []
  times = []

  print("Using features:", selected_features)
  for test_image in test_images:
    image = imread(test_image, apply_tresh=False)
    start_time = time.time()
    try:
      assert image is not None, f"{test_image} couldn't be read."
      image = preprocess(image)
      features = {}
      for feature in selected_features:
        features[feature] = run_feature_extraction(image, feature)
      prediction = clf.predict(features, use_probs=True)
      results.append(str(round(prediction)))
      decision = 'he/him' if round(prediction) else 'she/her'
      certainty = f'{abs(prediction - 0.5) * 200:.2f}%'
    except Exception as e:
      print(e)
      decision, certainty = 'they/them', 'unknown'
      results.append('-1')
    time_taken = f'{time.time() - start_time:.2f}'
    times.append(time_taken)
    if args.verbose: print(f"{test_image} identifies as {decision} with {certainty} certainty - took {time_taken}s")

  with open(os.path.join(args.outputdir, 'results.txt'), 'w') as results_file:
    results_file.write('\n'.join(results))

  with open(os.path.join(args.outputdir, 'times.txt'), 'w') as times_file:
    times_file.write('\n'.join(times))

if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  main()
