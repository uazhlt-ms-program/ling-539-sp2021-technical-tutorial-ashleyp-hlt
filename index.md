# **Building a Training Set with Snorkel**
This tutorial provides an overview of using Snorkel's labeling function to generate a training set of labels from raw, unlabeled data. The data we are working with in this example consists of google reviews of the Grand Canyon; obtained via the Apify web scraper. The resulting training set can be used to classify reviews as being informative or not.

## What is Snorkel?
Snorkel is a python package for building and managing training datasets, *programmatically*. It provides operations for labeling, transforming, and slicing data, however the focus of this tutorial is labeling. To replace the long tedious process of manually labeling a dataset by hand, Snorkel offers a weak supervision approach to apply labels using *labeling functions*. Labeling functions are programmatic rules and heuristics, created by the user, that Snorkel then models and combines into clean confidence-weighted labels. 

## Task: Information Detection
The goal behind building this training set is to label the google reviews as either ```STUFF``` or ```FLUFF``` to indicate whether the review provides useful information or not.
* ```STUFF```: contains information on topics that may be relevant to a future visitor, such as parking, scenic tours, facilities, food, etc.
  * ![Image of STUFF1] https://github.com/uazhlt-ms-program/technical-tutorial-ashleyp-hlt/blob/main/ScreenShots/STUFF1.png)
  * ![Image of STUFF2] (/ScreenShots/STUFF2.png)
  * ![Image of STUFF3] (/ScreenShots/STUFF3.png)

* ```FLUFF```: contains opinions and/or descriptions that do not provide any useful information.  
  * ![Image of FLUFF1] (/ScreenShots/FLUFF1.png)
  * ![Image of FLUFF2] (/ScreenShots/FLUFF2.png)
  * ![Image of FLUFF3] (/ScreenShots/FLUFF3.png)

## Step 1: Loading Data
As previously mentioned, the data for this tutorial was scraped from google reviews on the web and exported as an xml file. ElementTree was used to parse the file and extract each review and it's published-at date. Out of the 300 reviews collected, the first 200 were added to a dictionary designated as the training set, and the remaining 100 made up the test set. Snorkel works well with various types of DataFrames and provides native support for several sturctures, so the dictionaries were then converted to Pandas DataFrames.

```python 
import xml.etree.ElementTree as ET
import pandas as pd

rev_data = open('data.xml', 'r').read()
root = ET.XML(rev_data)

reviews = []
dates = []
for text in root.iter('text'):
    reviews.append(text.text)   
for date in root.iter('publishedAtDate'):
    dates.append(date.text)

dtrain = {}
for key in reviews[:200]:
    for value in dates[:200]:
        dtrain[key] = value     
dtest = {}
for key in reviews[200:]:
    for value in dates[200:]:
        dtest[key] = value  

dfr_train = pd.Series(dtrain).to_frame()
dfr_test = pd.Series(dtest).to_frame() 
```
## Step 2: Writing Labeling Functions (LFs)
In the task at hand, the goal is to label reviews as either ```STUFF``` or ```FLUFF```. A third label ```ABSTAIN``` exists for when a data point cannot be labeled.
```python
STUFF = 1
FLUFF = 0
ABSTAIN = -1
```
