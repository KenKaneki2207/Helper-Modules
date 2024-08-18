import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf

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


def create_confusion_matrix(test_data, model, shape, color=plt.cm.Blues):

  '''
  Pre-requisite : confusion_matrix, ConfusionMatrixDisplay functions from sklearn.metrics
    'from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay'

  Inputs :  test_data --> test_data object,
            model --> Model used for prediction.
            shape --> shape of the figure.
            color --> plt.cm.Blues(default, you can choose other colors from 
            https://matplotlib.org/stable/users/explain/colors/colormaps.html)

  Outputs : Display's a besutiful confusion matrix.

  '''
  
  # Test Labels will contain the true values
  display_labels = test_data.class_names

  test_labels = []
  for _,labels in test_data.unbatch():
    test_labels.append(tf.argmax(labels).numpy())
  print('True Labels :\n', test_labels[:10])
  print('The shape of test_labels :', len(test_labels))

  # Now lets find the predicted values
  pred_labels = []
  preds = model.predict(test_data)
  for pred in preds:
    pred_labels.append(tf.argmax(pred).numpy())
  print('Predicted Labels :\n', pred_labels[:10])
  print('The shape of predicted_labels :', len(pred_labels))

  # Create Confusion Matrix
  cm = confusion_matrix(y_true=test_labels, y_pred=pred_labels)

  # Display the confusion matrix
  fig, ax = plt.subplots(figsize=shape)
  ConfusionMatrixDisplay(cm, display_labels=display_labels).plot(ax=ax, cmap=color, xticks_rotation='vertical')


def compare_model_results(result_1, result_2, label_1, label_2):

  """
  Description : Returns the comparison of the evaluation metrics of the 2 models.

  Parameters : 
  result_1 --> Results of the model 1 stored in dictionary format.

  result_2 --> Results of the model 2 stored in dictionary format

  label_1 --> label for the model 1

  label_2 --> label for the model 2
  """
  X = list(result_1.keys())
  model_0 = list(result_1.values())
  model_1 = list(result_2.values())

  X_axis = np.arange(len(X)) 
  print(X_axis)

  plt.bar(X_axis - 0.2 , model_0, 0.4, label = label_1) 
  plt.bar(X_axis + 0.2 , model_1, 0.4, label = label_2) 

  plt.xticks(X_axis, X) 
  plt.xlabel("Evaluation Metrics") 
  plt.ylabel("Evaluation Results") 
  plt.title("Results for each model") 
  plt.legend(loc="center") 
  plt.show() 
