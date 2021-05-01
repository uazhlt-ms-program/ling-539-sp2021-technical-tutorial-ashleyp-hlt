# **Building a Training Set with Snorkel**
This tutorial provides an overview of using Snorkel's labeling function to generate a training set of labels from raw, unlabeled data. The [data](https://github.com/uazhlt-ms-program/technical-tutorial-ashleyp-hlt/blob/main/data.xml) we are working with in this example consists of google reviews of the Grand Canyon; obtained via the Apify web scraper. The resulting training set can be used to classify reviews as being informative or not.

## What is Snorkel?
Snorkel is a python package for building and managing training datasets, *programmatically*. It provides operations for labeling, transforming, and slicing data, however the focus of this tutorial is labeling. To replace the long tedious process of manually labeling a dataset by hand, Snorkel offers a weak supervision approach to apply labels using *labeling functions*. Labeling functions are programmatic rules and heuristics, created by the user, that Snorkel then models and combines into clean confidence-weighted labels. 

## Task: Information Detection
The goal behind building this training set is to label the google reviews as either ```STUFF``` or ```FLUFF``` to indicate whether the review provides useful information or not.
* ```STUFF```: contains information on topics that may be relevant to a future visitor, such as parking, scenic tours, facilities, food, etc.
   ![Image of STUFF1](/ScreenShots/STUFF1.png)
   ![Image of STUFF2](/ScreenShots/STUFF2.png)
   ![Image of STUFF3](/ScreenShots/STUFF3.png)

* ```FLUFF```: contains opinions and/or descriptions that do not provide any useful information.  
   ![Image of FLUFF1](/ScreenShots/FLUFF1.png)
   ![Image of FLUFF2](/ScreenShots/FLUFF2.png)
   ![Image of FLUFF3](/ScreenShots/FLUFF3.png)

## Step 1: Loading Data
As previously mentioned, the data for this tutorial was scraped from google reviews on the web and exported as an xml file. ElementTree was used to parse the file and extract each review and it's published-at date. Out of the 300 reviews collected, the first 200 were added to a dictionary designated as the training set, and the remaining 100 made up the test set. Snorkel works well with various types of DataFrames and provides native support for several structures, so the dictionaries were then converted to Pandas DataFrames.

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
for key in reviews:
    for value in dates:
        dtrain[key] = value     

dfr_train = pd.Series(dtrain).to_frame()
```
## Step 2: Writing Labeling Functions (LFs)
In the task at hand, the goal is to label reviews as either ```STUFF``` or ```FLUFF```. A third label ```ABSTAIN``` exists for when a data point cannot be labeled.
```python
STUFF = 1
FLUFF = 0
ABSTAIN = -1
```
Snorkel supports numerous different types of LFs such as keyword searches, pattern matching, third-party models, distant supervision, and crowdworker labels. It is up to the user to decide which types of LFs would be beneficial, and how many should be created. In order to generate ideas for LFs, it is recommended to look at random data points from the training set and identify any class indicators.

Looking at the three reviews used as examples of ```STUFF```, we see that in the first one "Be careful" is followed by a warning about weather conditions. In the next review, "we suggest" is followed by a recommendation about bringing water and avoiding heat exhaustion. It also mentions an app that acts as a tour guide. The third review preludes information about getting into the park with "warning..." and also mentions the price of helicopter rides. From these reviews alone several keyword search LFs can be generated as well as a pattern match LF to find reviews that contain "$". The reviews above that were selected as examples of ```FLUFF``` show that this class is lacking when it comes to indicators. This was to be expected for our task because it is difficult to detect the absence of *something* when that something is not explicitly defined. However, these reviews do appear to be shorter than those that provide information, so a heuristic LF could be created based on the length of a review.

To actually write a LF, simply place the decorator ```@labeling_function()``` above a python function that returns a label.
```python
from snorkel.labeling import labeling_function
from snorkel.labeling import LFAnalysis
import re

@labeling_function()
def recommend(x):
    return STUFF if re.search("recommend|advice|advise|suggest|bring",str(x)) else ABSTAIN

@labeling_function()
def warning(x):
    return STUFF if re.search("warning|watch out|be sure to|plan for|be careful|avoid", str(x)) else ABSTAIN

@labeling_function()
def parking(x):
    return STUFF if re.search("parking",str(x)) else ABSTAIN

@labeling_function()
def tour(x):
    return STUFF if re.search("tour|helicopter|plane",str(x)) else ABSTAIN

@labeling_function()
def shuttle(x):
    return STUFF if re.search("shuttle[s]?|bus|bus stops",str(x)) else ABSTAIN

@labeling_function()
def bathroom(x):
    return STUFF if re.search("bathroom[s]?|facilities",str(x)) else ABSTAIN

@labeling_function()
def food(x):
    return STUFF if re.search("food|cafe|restaurant",str(x)) else ABSTAIN

@labeling_function()
def pricing(x):
    return STUFF if re.search("\$",str(x)) else ABSTAIN

@labeling_function()
def length(x):
    return FLUFF if len(str(x)) < 60 else ABSTAIN
```
## Step 3: Applying Label Functions
The Snorkel Labeling Package provides several LF appliers that can be imported to apply the developed LFs to the data points. For larger datasets, larger LF sets, and LFs that require more computation, it is recommended to use the Dask DataFrames or PySpark DataFrames with their respective appliers. Since the data points for this tutorial are formatted in a Pandas DataFrame, we will be using the PandasLFApplier. The parameter for this applier is a list of LFs and the ```.apply()``` method results in a label matrix. Each row represents a data point and each column corresponds to a LF.
```python
from snorkel.labeling import PandasLFApplier
lfs = [recommend, warning, parking, tour, shuttle, bathroom, food, pricing, length]
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(dfr_train)
print(L_train)
[[ 1 -1 -1 ...  1 -1 -1]
 [ 1 -1 -1 ... -1 -1 -1]
 [ 1 -1 -1 ... -1 -1 -1]
 ...
 [-1 -1 -1 ... -1 -1 -1]
 [-1 -1 -1 ... -1 -1 -1]
 [-1 -1 -1 ... -1 -1 -1]]
```
## Step 4: LF Analysis
The initial LFs that are created are known to be noisy and imperfect. Some data points may have multiple labels, some may have none, and some may have conflicting labels. Analysis of the LFs can be done using Snorkel's ```LFAnalysis```, provided in the labeling package. This tool has a variety of methods including ```lf_conflicts()```, ```lf_coverages()```, ```lf_overlaps()```, and ```lf_polarities()```. The ```lf_summary()``` method combines these outputs into an easy to read table. Polarity is the label assigned by the LF, and coverage is the percentage of data points labeled by the LF.
```python
from snorkel.labeling import LFAnalysis
LFAnalysis(L=L_train, lfs=lfs,).lf_summary()
```
![LFSummary](https://github.com/uazhlt-ms-program/technical-tutorial-ashleyp-hlt/blob/main/ScreenShots/LFsummary.png)

Looking at the table we can see that the ```length``` LF did not label any data points and therefore should be revised or deleted.
To see the total percentage of data points that were labeled we can use ```LFAnalysis(L=L_train).label_coverage()```. The output of that line is 0.354, indicating that 35.4% of data points were labeled on the initial run. The goal is not to label every data point, but to create a solid set of labels that can be used to train a classifier.

To check for false positives we can use

