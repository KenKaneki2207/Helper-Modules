import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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


def plot_history(history):

  '''
  Prerequisite : matplotlib.pyplot as plt

  Input : history --> history variable.

  Output : Loss and Accuracy curve for training and validation.
  '''

  t_loss = history.history['loss']
  v_loss = history.history['val_loss']

  t_acc = history.history['accuracy']
  v_acc = history.history['val_accuracy']

  plt.title('Loss Curve')
  plt.plot(t_loss)
  plt.plot(v_loss)
  plt.figure()

  plt.title('Accuracy Curve')
  plt.plot(t_acc)
  plt.plot(v_acc)


def confusion_matrix_graph(cm, labels, shape, color=plt.cm.Blues):

  '''
  Pre-requisite : confusion_matrix, ConfusionMatrixDisplay functions from sklearn.metrics
    'from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay'

  Inputs :  cm --> confusion matrix,
            labels --> labels for the x-axis
            shape --> shape of the figure.
            cmap --> plt.cm.Blues(default, you can choose other colors from 
            https://matplotlib.org/stable/users/explain/colors/colormaps.html)

  Outputs : Display's a besutiful confusion matrix.

  '''

  cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)
  
  fig, ax = plt.subplots(figsize=shape)
  cm_display.plot(ax=ax, cmap=color, xticks_rotation='vertical')
