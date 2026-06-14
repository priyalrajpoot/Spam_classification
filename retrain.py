import pandas as pd
import numpy as np
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

# Ensure nltk packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# 1. Load Data
print("Loading data...")
df = pd.read_csv('spam.csv', encoding='latin-1')

# 2. Data Cleaning
# drop last 3 cols (Unnamed: 2, 3, 4)
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True, errors='ignore')
# rename cols
df.rename(columns={'v1':'target','v2':'text'}, inplace=True)
# label encode target
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
# remove duplicates
df = df.drop_duplicates(keep='first')

# 3. Preprocessing
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

print("Preprocessing text (this might take a minute)...")
df['transformed_text'] = df['text'].apply(transform_text)

# 4. Vectorize & Model
print("Vectorizing...")
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

print("Training model...")
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Evaluation
y_pred = mnb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
print(f"Model trained successfully under scikit-learn 1.7.2!")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")

# Save vectorizer and model
print("Saving vectorizer.pkl and model.pkl...")
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(mnb, f)

print("Done! New pickles are ready.")
