import numpy as np
import tensorflow.python.keras
from tensorflow.python.keras import layers,models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import cv2

import os
import shutil


#-----------------------------------------------------------
#data arrangement
#-----------------------------------------------------------
categories=["normal","turmor"]
nb_classes=len(categories)

f=np.load("datasets.npz")
testImg, testLabel = f["testimg"], f["testlabel"]
trainImg, trainLabel = f["trainimg"], f["trainlabel"]
valImg, valLabel = f["valimg"], f["vallabel"]
f.close()

test_img = testImg.astype("float")/255.0
train_img = trainImg.astype("float")/255.0
val_img = valImg.astype("float")/255.0

test_label = np_utils.to_categorical(testLabel,nb_classes)
train_label = np_utils.to_categorical(trainLabel,nb_classes)
val_label = np_utils.to_categorical(valLabel,nb_classes)

print("train:",len(train_img),"/",len(train_label))
print("val:",len(val_img),"/",len(val_label))
print("test:",len(test_img),"/",len(test_label))

#------------------------------------------------------------
#read last model
#------------------------------------------------------------
model = models.load_model("/home/akiba/newcomer/assignment/2_MedicalImageClassification/model.h5")
model.summary()

#------------------------------------------------------------
#test
#------------------------------------------------------------
loss, acc = model.evaluate(test_img,test_label)

print("loss:",loss)
print("acc:",acc)

#------------------------------------------------------------
#confusion matrix
#------------------------------------------------------------
predict_classes = model.predict_classes(test_img)
true_classes = testLabel

print(true_classes)
print(predict_classes)

print(confusion_matrix(true_classes, predict_classes))

def print_cmx(y_true, y_pred):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
 
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cmx, annot=True, fmt='g' ,square = True)
    plt.show()
 
print_cmx(true_classes, predict_classes)

#------------------------------------------------------------
#classify images based on confusion matrix
#------------------------------------------------------------
dPath = "/home/akiba/newcomer/assignment/2_MedicalImageClassification/Dataset"
dClass = ["test","train","val"]
dLabel = ["0","1"]
TP = []
FP = []
FN = []
TN = []

shutil.rmtree(dPath+"/TP")
shutil.rmtree(dPath+"/FP")
shutil.rmtree(dPath+"/FN")
shutil.rmtree(dPath+"/TN")

os.mkdir(dPath+"/TP")
os.mkdir(dPath+"/FP")
os.mkdir(dPath+"/FN")
os.mkdir(dPath+"/TN")

list0 = os.listdir(dPath+"/test/0")
list1 = os.listdir(dPath+"/test/1")
list01 =list0 + list1

print(list01[0])

for i in range(len(list01)):
    if (true_classes[i] == 0) & (predict_classes[i] == 0):
            TN.append(list01[i])
    elif(true_classes[i] == 1) & (predict_classes[i] == 1):
            TP.append(list01[i])
    elif(true_classes[i] == 0) & (predict_classes[i] == 1):
            FP.append(list01[i])
    else:
            FN.append(list01[i])

print("number of TN:",len(TN))
print("number of TP:",len(TP))
print("number of FN:",len(FN))
print("number of FP:",len(FP))

for i in list0:
       im = cv2.imread(dPath+"/test/0/"+i)
       if i in TN:
              cv2.imwrite(dPath+"/TN/"+i,im)
       else:
              cv2.imwrite(dPath+"/FP/"+i,im)

for i in list1:
       im = cv2.imread(dPath+"/test/1/"+i)
       if i in TP:
              cv2.imwrite(dPath+"/TP/"+i,im)
       else:
              cv2.imwrite(dPath+"/FN/"+i,im)