# Background
In this capstone project, I build an image classifier to detect whether a tire texture is normal or cracked(Oxidized).

The image dataset was taken from this link : https://www.kaggle.com/jehanbhathena/tire-texture-image-recognition.

There are  a total of 1028 images that consists of  703 training images and 325 testing images. From the 703 training images, 327 are cracked tires while 376 are normal tires. Meanwhile from the 325 testing images, 210 are cracked tires and 115 are normal tires. 

The size of the images ranges from 224x224 to 3024x3024. 

# Model Selection and Paramater Tuning
I use two algorithm for this problem which is CNN and Transfer Learning using Xception. I train the neural network on training data and use the testing data for evaluation. The total number of training and testing data is the same as the source. The image dimension input for this step is 150x150. I did some parameter tuning on Transfer Learning. The best model was transfer learning with the following configuration : 
>- learning rate = 0.0001 
>- size of first dense layer = 256 
>- dropout = 0.2 

Then I train the model using a input image 224x224 (the smallest dimension in training and testing combined) and save the best model with callback in HDF5 format(.h5). 

The saved model is loaded back to notebook and the final evaluation was :
>- val_loss : 0.5474080443382263
>- val_accuracy :  0.8246153593063354

Then I saved the model in SavedModel format so that I can serve the model with TensorFlow Serving. 

The code for training the final model can be reproduced in train.py. When loading the model, you can change the HDF5 file name to the file name that is saved in your computer by callbacks. Then the loaded model will be saved as SavedModel format.

#  Serving the model with TensorFlow Serving, using flask as gateway, and deploying it locally via docker:
 Before we serve the model, we use : 
 >- saved_model_cli show --dir {model_name} --all 
 
 to get the signature, input and output to be used in gateway.py. 
  
  I use tensorflow serving for serving the model and flask as the gateway between the user and the model. I build two images :
  >- gateway image
  >- tensorflow serving image

  In tensorflow serving Dockerfile which is image-model.dockerfile, you can change the directory of the model.


  I use pipenv to create the Pipfile.lock and Pipfile file for the python environment in the gateway image.
  >- pipenv install grpcio==1.42.0 flask gunicorn keras-image-helper tensorflow-protobuf==2.7.0
  
  The images was build using Dockerfile.
  Command to build the image : 
  >- tensorflow serving image : docker build -t {image_name} -f image-model.dockerfile .
  >- gateway image : docker build -t {image_name} -f image-gateway.dockerfile .

After we build those two images, we can use docker-compose to connect those images in one network. First we need to create a docker-compose.yaml file to configure what images are we going to connect. 

In docker-compose.yaml, you can change the image name.

Command for docker-compose : 
  >- to launch the program : docker-compose up (make sure the command is executed in the same directory as the docker-compose.yaml file)
  >- to shut down the program : docker-compose down 

I test the model by using running test.py.
