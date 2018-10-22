# Tensor Flow Implementation
This project was done for a course, this one geared towards Machine Learning.  These classes were separated into Trains and Cars, specifically. Note however, more categories can be added.

For installation, please follow instructions on https://www.tensorflow.org/install/pip (this installation uses pip, which it assumes you have not installed yet).  Note that for this design, you will also need tensorflow_hub.  

There are a multitude of dataset sites offered online.  I used Kaggle, although there are plenty of ways to use this.  To install, follow installations instructions on https://github.com/Kaggle/kaggle-api.  You will need to export your username and key.  You must create a Kaggle account, and request an API key if you don't already have one.
After this, you can go to Kaggle and search for datasets. I used the famous Stanford cars dataset.  I also used a trains dataset. To download the dataset, have a terminal open to the directory you prefer, and copy the clipboard command that should be on the Kaggle site for the dataset you want.

In order to implement my ML design, I used Tensor Flow licensed programming, and have properly cited as such.  The tf_retrain.py is based on the retrain.py found here: 'https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py/'.  This trains a model to recognize against multiple classes, in which training data is pre-sorted into directories within the working directory.  For my design, I had a cars directory and train directory in the same folder as tf_retrain.py.  
Note that retrain takes a significant amount of time, so I significantly reduced the sample size of the Stanford dataset to only 2000 pictures.  The dataset of trains only had ~1000 images.
To run retrain, run the following:
python3 tf_retrain.py --image_dir /PATH/TO/MY/TRAINING/DATA

The tf_label_image.py tests an image, giving confidence of the result. To run, do the following:

python tf_label_image.py \
--graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
--input_layer=Placeholder \
--output_layer=final_result \
--image=/PATH/TO/TESTING/TEST.jpg

 
Running these tests, with both test files, here are the results:

testCar.jpg:
cars 0.99988115
train 0.00011882853

testCarBlurry.jpg
cars 0.74984735
train 0.25015265

testTrain.jpg:
train 0.9996586
cars 0.00034138633

testTrainBlurry.jpg:
train 0.9999548
cars 4.515491e-05

One may refer to the pictures in this repo to see for oneself how well these results work.  However, I will state that the results are accurate, never falsely identifying in these particular trials. 

This seems to be an elegant design, but there are some issues of note.  Mainly, the tmp folder gets cleared frequently, so the learning data could be cleared, meaning that there will have to be retraining.  Improvements would be to tweak this dense code such that it outputs to a more convenient location.


# System Comparison
