def walk_directory(path):
  '''
  Prerequisities Libraries : import os, before using this function.

  Input : path --> path to the directory

  Output : All the sub-directories and filenames in the path provided.
  
  '''

  for dirpath, dirnames, filenames in os.walk(path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")


def random_images(path, class_name=None):
  '''
  Prerequisite Libraries: os, random, matplotlib.pyplot and  matplotlib.image
  ""import random
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    import os""
  

  Input : path --> path to the directory,
          class_name = None(by default), can provide the name of a specific directory in given path.

  Output : random iamge from the path directory.

  '''

  if class_name == None:
    classes = random.choice(os.listdir(path))
  else:
    classes = class_name
  img = random.choice(os.listdir(f"{path}/{classes}"))
  title = classes

  plt.imshow(imread(f"{path}/{title}/{img}"))
  plt.axis("off")
  plt.title(title)
