
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

df = pd.read_csv("https://raw.githubusercontent.com/SmallLion/Python-Projects/main/Spam-detection/spam.csv")
df = df.filter(items=["v1","v2"])

# Customize the column names
df = df.rename(columns= {"v1":"Label", "v2":"EmailText"})

# Change the target name value of "normal". This is not mandatory, and only for human understanding 
df.Label = df.Label.replace("ham", "normal")

# Split the dataframe and assign train and test sets and use 0.20 as 20% for test dataset

x = df["EmailText"]
y = df["Label"]

# Split the test to avoid either overfitting or underfitting.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Instantiate the CountVectorizer for tokenization and fit_transform the test data

cv = CountVectorizer()
features = cv.fit_transform(x_train)

# Instantiate the Model and fit the tokenized train dataset and the y_train (target value)

model = svm.SVC()
model.fit(features, y_train)

# Use the test datasets to test the model for accuracy score. Recall to transform the x_test to tokens

features_test = cv.transform(x_test)
print("Accuracy score: {}".format(model.score(features_test, y_test)))