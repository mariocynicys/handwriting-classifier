#!env python
import argparse
from tkinter import filedialog, Tk


def glob_it(directory: str):
    '''Returns all the files inside a directory and their subdirectories, infinitely.'''
    from os import listdir
    from os.path import abspath, join, isfile, isdir
    directory = abspath(directory)
    dirlisting = listdir(directory)
    files = [join(directory, file) for file in dirlisting
                if isfile(join(directory, file))]
    dircs = [join(directory, dirc) for dirc in dirlisting
                if isdir(join(directory, dirc))]
    for dirc in dircs:
        files.extend(glob_it(dirc))
    return files


def main():
  parser = argparse.ArgumentParser(description='A male/female handwriting classifier.')
  parser.add_argument('-i', '--datadir',
                      help='The path to the input directory.  This directory will be searched recursively for images.')
  args = parser.parse_args()

  if args.datadir is not None:
    datadir = args.datadir
  else:
    Tk().withdraw()
    datadir = filedialog.askdirectory(title='Select an Input Direcotry', mustexist=True)

  images = glob_it(datadir)


if __name__ == '__main__':
  try:
    main()
  except Exception as e:
    print('Error: ', e)
    exit(1)
