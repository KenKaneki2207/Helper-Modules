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


from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def evaluation(true_labels, pred_labels):

  '''
  Description : 
    It returns the accuracy, f1, precision, recall in dictionary format.
  accuracy --> The percentage of correctly predicted instances out of the total instances.
  f1 --> The harmonic mean of precision and recall. It is useful when you need a balance between precision and recall.
  precision --> The number of correctly predicted positive observations divided by the total predicted positives. It indicates how many of the predicted positives are actually positive.
  recall --> The number of correctly predicted positive observations divided by the total actual positives. It indicates how many of the actual positives are captured by the model.


  Prerequisite : from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


  Input : 
    true_labels --> true labels used for evaluations, 
    pred_labels --> Predictions made by your model

  Output : 
    It returns the accuracy, f1, precision, recall in dictionary format.    
  
  '''

  return {'accuracy' : accuracy_score(true_labels, pred_labels), 
          'f1' : f1_score(true_labels, pred_labels),
          'precision' : precision_score(true_labels, pred_labels),
          'recall' : recall_score(true_labels, pred_labels)}


import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def text_preprocessing(text):
    
    '''
    Description : Text_preprocessing : 
                lowering, removing characteres,urls, tokenize the text, removing stopwords
    
    Prerequiste : import re
                  from nltk.corpus import stopwords
                  from nltk.tokenize import word_tokenize
                  
    Input : text
    
    Ouptut : transformed text
    '''
    
    #lower the text
    text = text.lower()
    
    # Removing all the characters except strings
    text = re.sub('[^a-z ]','', text)
    
    # Remove the urls
    text = re.sub('http\S+|www.\S+','',text)
    
    # Tokenizing
    words = word_tokenize(text)
    
    # removing stopwords
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    
    # concating the filtered words
    text = ' '.join([word for word in filtered_words])
    
    return text


# Abbreviations dictionary
abbreviations = {
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    'r' : 'are',
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}

def expand_abbrevations(text):

  '''
  Description : It expands the abbreviations to its original form.

  Prerequisites : None

  Input : Abrrevated texts

  Output : Expansion of abbreviations.
  '''
    keys = abbreviations.keys()
    words = text.split()
    for i in range(len(words)):
        if words[i] in keys:
            words[i] = abbreviations[words[i]]
    
    # concating the words
    text = ' '.join([word for word in words])
    
    return text
