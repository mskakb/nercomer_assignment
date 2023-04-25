import numpy as np
import tensorflow.python.keras
from tensorflow.python.keras import layers,models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.utils import np_utils
import matplotlib.pyplot as plt

#-----------------------------------------------------------
#data arrangement
#-----------------------------------------------------------
categories=["tumor","normal"]
nb_classes=len(categories)

f=np.load("datasets.npz")
testImg, testLabel = f["testimg"], f["testlabel"]
trainImg, trainLabel = f["trainimg"], f["trainlabel"]
valImg, valLabel = f["valimg"], f["vallabel"]
f.close()

test_img = testImg.astype("float")/255.0
train_img = trainImg.astype("float")/255.0
val_img = valImg.astype("float")/255.0

print(testLabel)

test_label = np_utils.to_categorical(testLabel,nb_classes)
train_label = np_utils.to_categorical(trainLabel,nb_classes)
val_label = np_utils.to_categorical(valLabel,nb_classes)

print(test_label)


print("train:",len(train_img),"/",len(train_label))
print("val:",len(val_img),"/",len(val_label))
print("test:",len(test_img),"/",len(test_label))

#-----------------------------------------------------------
#create CNN
#-----------------------------------------------------------
model=models.Sequential()
model.add(layers.Conv2D(128,(3,3),activation="relu",input_shape=(224,224,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(32,(3,3),activation="relu"))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Dropout(0.1))
model.add(layers.Flatten())
model.add(layers.Dense(128,activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(nb_classes,activation="softmax"))
model.summary()

json_string=model.to_json()
open("train.json","w").write(json_string)

#-----------------------------------------------------------
#train
#-----------------------------------------------------------
NUM_EPOCHS =10
BATCH_SIZE =64

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["acc"])
history = model.fit(train_img,train_label,batch_size=BATCH_SIZE,epochs=NUM_EPOCHS,validation_data=(val_img,val_label),shuffle=True)

model.save_weights("train.hdf5")
model.save("model.h5")

#-----------------------------------------------------------
#figure
#-----------------------------------------------------------
fig, (axL, axR) = plt.subplots(ncols = 2, figsize = (20, 8))

axL.set_title("model loss")
axL.set_xlabel("epoch")
axL.set_ylabel("loss")

axR.set_title("model accuracy")
axR.set_xlabel("epoch")
axR.set_ylabel("accuracy")

axR.plot(range(1,NUM_EPOCHS+1),history.history["acc"],"-o")
axR.plot(range(1,NUM_EPOCHS+1),history.history["val_acc"],"-o")
axR.grid()
axR.legend(['acc','val_acc'],loc='best')

axL.plot(range(1,NUM_EPOCHS+1),history.history["loss"],"-o")
axL.plot(range(1,NUM_EPOCHS+1),history.history["val_loss"],"-o")
axL.grid()
axL.legend(["loss","val_loss"],loc="best")

plt.show()