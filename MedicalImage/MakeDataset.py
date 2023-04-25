from keras.preprocessing.image import array_to_img,img_to_array,load_img
import numpy as np
import os

testImg = []
testLabel = []

trainImg = []
trainLabel =[]

valImg = []
valLabel = []

dPath = "/home/akiba/newcomer/assignment/2_MedicalImageClassification/Dataset"
dClass = ["test","train","val"]
dLabel = ["0","1"]

for i in dClass:
    for j in dLabel:
        if j=="0":
            label = 0
        elif j=="1":
            label = 1
        list = os.listdir(dPath+"/"+i+"/"+j)
        for target in list:
            image = dPath+"/"+i+"/"+j+"/"+target
            temp_img = load_img(image)
            temp_img_array = img_to_array(temp_img)
            if i=="test":
                testImg.append(temp_img_array)
                testLabel.append(label)
            elif i=="train":
                trainImg.append(temp_img_array)
                trainLabel.append(label)
            elif i=="val":
                valImg.append(temp_img_array)
                valLabel.append(label)

print("train:",len(trainImg),"/",len(trainLabel))
print("val:",len(valImg),"/",len(valLabel))
print("test:",len(testImg),"/",len(testLabel))

img = load_img("/home/akiba/newcomer/assignment/2_MedicalImageClassification/Dataset/train/0/15396.png")
img_array = img_to_array(img)
print(img_array.shape)

np.savez("datasets.npz",testimg=testImg,testlabel=testLabel,trainimg=trainImg,trainlabel=trainLabel,valimg=valImg,vallabel=valLabel)   