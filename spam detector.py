
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

df = pd.read_csv("https://raw.githubusercontent.com/SmallLion/Python-Projects/main/Spam-detection/spam.csv")
df = df.filter(items=["v1","v2"])