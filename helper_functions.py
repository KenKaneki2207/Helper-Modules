def walk_directory(path):
  '''
  Prerequisities Libraries : import os, before using this function.

  Input : path --> path to the directory

  Output : All the sub-directories and filenames in the path provided.
  
  '''

  for dirpath, dirnames, filenames in os.walk(path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")
