# Spam_detection
![image](https://user-images.githubusercontent.com/31643510/200195666-3e190bce-8b04-400a-8d1a-26a9726a03bb.png)
 <br>
This is a Machine Learning Projects that aims at predicting the spam in a email contents. 
# Libraries imported:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
```
Here, I built and ran Support Vector Machine Algorithm model, comparing the predictions against the actual output. 
Finally, I tested the model with the countVectorizer from the feature_extraction.text of the Sci-kit Learn model.
This was used to tokenize the dataset

The accuracy score for this little spam detector was as high at 98%.
