# **Building a Training Set with Snorkel**
This tutorial provides an overview of using Snorkel's labeling function to generate a training set of labels from raw, unlabeled data. The data we are working with in this example consists of google reviews of the Grand Canyon; obtained via the Apify web scraper. The resulting training set can be used to classify reviews as being informative or not.

## What is Snorkel?
Snorkel is a python package for building and managing training datasets, *programmatically*. It provides opperations for labeling, transforming, and slicing data, however the focus of this tutorial is labeling. To replace the long tedious process of manually labeling a dataset by hand, Snorkel offers a weak supervision approach to apply labels using *labeling functions*. Labeling functions are programmatic rules and heuristics, created by the user, that Snorkel then models and combines into clean confidence-weighted labels. 

## Task: Information Detection
The goal behind building this training set is to label the google reviews as either ```STUFF``` or ```FLUFF``` to indicate whether the review provides useful information or not.
* ```STUFF```: contains information on topics that may be relevant to a future traveler, such as parking, scenic tours, facilities, food, etc.
  * ![Image of STUFF1] (/Desktop/STUFF1.png)

* ```FLUFF```: contains opinions and/or descriptions that do not provide any useful information.  

## Step 1: Loading Data
 
