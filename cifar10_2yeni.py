import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers,losses,datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
import matplotlib.pyplot as plot
#%%
(train_images,train_labels),(test_images,test_labels)=datasets.cifar10.load_data()
plot.imshow(train_images[2],cmap="Greys") 
print("Label",train_labels[2])#deneme
#%%
boyut=32*32*3
train_images=train_images.reshape(-1,boyut)/255.0
test_images=test_images.reshape(-1,boyut)/255.0
#%%
model = Sequential([
Dense(boyut, activation=tf.nn.relu,input_shape=(boyut,),name="Inputs"),
Dense(boyut/2,activation=tf.nn.relu,input_shape=(boyut,),name="Hidden1"),
Dropout(0.2,name="Dropout1"),
Dense(boyut/4, activation=tf.nn.relu,
input_shape=(boyut/2,),name="Hidden2"),
Dropout(0.2,name="Dropout2"),
Dense(10, activation=tf.nn.softmax,name="Output")
])
model.compile(optimizer=optimizers.Adam(),loss=losses.sparse_categorical_crossentropy
,metrics=['accuracy'])
model.summary()
#%%

model.fit(train_images, train_labels, epochs = 10,validation_split=1,validation_data =(test_images,test_labels),verbose=1)#verbose=1 adım adım göstermekte
loss, acc = model.evaluate(test_images, test_labels)
print("Trained model, accuracy: {:5.2f}%, loss: {:3.5f}".format((100*acc),loss))
#%%  ne kadar doğru olduğu hakkında test etme(ilk 20 görsel için deneme)
sınıflar = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

for i in range(0,20):
    #plot.imshow(test_images[i].reshape(32, 32, 3),cmap='Greys')
    pred = model.predict(test_images[i].reshape(-1,boyut))
    print("Label :",sınıflar[test_labels[i][0]]," Predict :",sınıflar[pred.argmax()])