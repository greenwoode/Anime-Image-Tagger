# AnimeTagNet

## 1. Introduction

The objective of this project is to take an anime image and assign multiple tags per image, with the associated accuracies.
There are certain websites that display images with various tags to help clairify to the viewer of what they are looking at. Howver, this process can be tedious and we wanted to simplify it by using this specific dataset that already had the tags and write a model to make the process automatic and quicker. 

## 2. Selection of Data

The data we selected was from the following hyperlink : https://www.kaggle.com/datasets/mylesoneill/tagged-anime-illustrations. We took the 331k image dataset to limit training and testing time, we chose to use 40k images for the training and 10k images for the testing. If we could, we would have used 80% of the images for training and 20% for testing the models accuracy. Along with the images we also had 130k plus tags that we had to limit. We were able to cut it down to about 90k by blacklisting any non-letter or numerical characters. To limit this further we sorted through the tags and condensed the ones with one word into one file and ones with multiple words into another. With a goal of trying to limit the tags down to around 2k by using the most common tags from the one word file. 


## 3. Methods

Tools :

 - Github for communication and sharing
 - VS Code as IDE along with Jupyter
 - Numpy 
 - Pandas
 - CV2 (image handling/processing)
 - Tensorflow
   - Keras
  
Methods :

- Keras:
  - CNN
  - Pooling
  - Custom Batches
- One-hot encoding
- Multi-Label categorization

## 4. Results

After running the model several times with various images, we keep seeing the prediction accuracy staying around the 56% mark. The model is able to predict various labels associated with a given image, however none of the actual labels are in the specific image's predicted set. The most likely cause for this is that the prediction for the tags associated with the image are falling too far below the frequency mark for it to be considered in the prediction. 


## 5. Discussion

There are two possible options that we could have used to improve our neural networks accuracy. First we believe that if we expanded the list of tags to cover more of the dataset, we could have seen the right labels associated to each of the images that we used to test our model. Second, we think if we had limited our list of tags to be more aligned with the images in our test instead. Currently the list of tags being used can be applied to the entire image set, however with the limited dataset, not all the images are having the tags applying to them. 

## 6. Summary

The program created uses convolutional neural networks to determine the most accurate tags for a given photo. It is able to make predictions as to what tags should be associated with a given image, however, it isn't always as accurate as we were hoping for. 

## 7. References 

 - Ethan Greenwood | greenwoode1@wit.edu
 - Kevin Mortimer | mortimerk1@wit.edu
 - Julia Winsper | winsperj@wit.edu
