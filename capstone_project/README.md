# Background
In this capstone project, I build an image classfier to detect whether a tire texture is normal or cracked(Oxidized). Cracked tires can poses a safety risk if not taken care properly. While this cracking can be a common sign of aging in rubber tires, it is also a sign of potential trouble that drivers need to take seriously. (https://www.tireoutlet.com/blog/3005/cracked-tires-when-theyre-unsafe/)

The image dataset was taken from this link : https://www.kaggle.com/jehanbhathena/tire-texture-image-recognition.

There are  a total of 1028 images that consists of  703 training images and 325 testing images. From the 703 training images, 327 are cracked tires while 376 are normal tires. Meanwhile from the 325 testing images, 210 are cracked tires and 115 are normal tires. 

The size of the images ranges from 224x224 to 3024x3024. 

# Model Selection and paramater Tuning
I use two algorithm for this problem which is CNN and Transfer Learning using Xception. I train the neural network on training data and use the testing data for evaluation. The total number of training and testing data is the same as the source. I did some parameter tuning on Transfer Learning. The best model was transfer learning with the following configuration : 
>- learning rate = 0.0001 
>- size of first dense layer = 256 
>- dropout = 0.2 

The final evaluation was :
>- val_loss : 0.5474080443382263
>- val_accuracy :  0.8246153593063354
