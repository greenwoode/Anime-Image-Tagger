# AnimeTagNet

## 1. Introduction

The objective of this project is to take an anime image and assign multiple tags per image, with the associated accuracies.
There are certain websites that display images with various tags to help clairify to the viewer of what they are looking at. Howver, this process can be tedious and we wanted to simplify it by using this specific dataset that already had the tags and write a model to make the process automatic and quicker. 

## 2. Selection of Data

The data we selected was from the following hyperlink : https://www.kaggle.com/datasets/mylesoneill/tagged-anime-illustrations. We took the 331k image dataset and used 80% for training the model and 20% for testing the model and its accuracy. Along with the images we also had 130k plus tags that we had to limit. We were able to cut it down to about 90k by blacklisting any non-letter or numerical characters. To limit this further we sorted through the tags and condensed the ones with one word into one file and ones with multiple words into another. With a goal of trying to limit the tags down to around 2k by using the most common most tags from the one word file. 


## 3. Methods

Tools :

 - Github for communication and sharing
 - VS Code as IDE along with Jupyter
 - Numpy 

Methods :

- Convolutional Neural Netowrk
- 
- 

## 4. Results


## 5. Discussion


## 6. Summary

The program created uses convolutional neural networks to determine the most accurate tags for a given photo. 

## 7. References 

 - Ethan Greenwood | greenwoode1@wit.edu
 - Kevin Mortimer | mortimerk1@wit.edu
 - Julia Winsper | winsperj@wit.edu
