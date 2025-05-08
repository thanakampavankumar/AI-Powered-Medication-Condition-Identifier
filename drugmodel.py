#!/usr/bin/env python
# coding: utf-8

# In[254]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder



# ## Loading the Dataset
# 
# We load the drug review dataset from an Excel file using `pandas.read_excel()`.

# In[255]:


main=pd.read_excel("drugsCom_raw.xlsx")


# We filter the dataset to include only reviews related to the following conditions:
# - Depression
# - High Blood Pressure
# - Diabetes, Type 2

# In[256]:


df=main[main["condition"].isin(["Depression","High Blood Pressure","Diabetes, Type 2"])]


# In[257]:


df.head()


# In[258]:


df.groupby("drugName")["rating"].mean()


# average rating for each condition

# # Apply both stopword removal and special character cleaning 

# In[259]:


# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Setup tools
stop = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def clean_review_with_stop_and_stem(text):
    # Convert to string and lowercase
    text = str(text).lower()

    # Remove special characters and stopwords
    words = [word for word in text.split() if word not in stop]
    text_no_specials = re.sub(r'[^a-z0-9\s]', '', ' '.join(words))

    # Lemmatize and Stem each word
    processed_words = []
    for word in text_no_specials.split():
        # Lemmatize the word
        lemma = lemmatizer.lemmatize(word)
        # Stem the lemmatized word
        stemmed = stemmer.stem(lemma)
        processed_words.append(stemmed)

    return ' '.join(processed_words)

# Apply to DataFrame column
df['cleaned'] = df['review'].apply(clean_review_with_stop_and_stem)


# join all reviews into a single text and convert to lowercase

# In[260]:


all_reviews_text = ' '.join(df['cleaned']).lower()
words = re.findall(r'\b\w+\b', all_reviews_text)
words = [word for word in words if word not in stop]

word_counts = Counter(words)
top_common_words = word_counts.most_common(20)

top_words_df = pd.DataFrame(top_common_words, columns=['Word', 'Frequency'])


# Top 20 Most Frequent Words (Overall)

# Filtering Rows Containing the Number "39" in Cleaned Text

# In[261]:


df[df['cleaned'].str.contains(r'\b39\b', regex=True, na=False)].head()


# In[262]:


df[df['cleaned'].str.contains(r'\b39\b', regex=True, na=False)].shape


# After cleaning the text, some entries may have "I039m" or "39 years". We remove "I039m" because it's a mistake, but keep "39 years" because it’s useful.

# In[263]:


df['cleaned'] = df['cleaned'].apply(lambda x: ' '.join([word for word in str(x).split() if not (('39' in word) and word != '39' and not word.isdigit())]))


# After removing I039m

# In[264]:


df[df['cleaned'].str.contains(r'\b39\b', regex=True, na=False)].head()


# In[265]:


reviews_text_combined = ' '.join(df['cleaned']).lower()
extracted_words = re.findall(r'\b\w+\b', reviews_text_combined)
filtered_words = [word for word in extracted_words if word not in stop]

word_frequency = Counter(filtered_words)
top_words = word_frequency.most_common(20)

top_words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])


# In[266]:


# pip install vaderSentiment


# Classifying Reviews as Positive or Negative:

# In[267]:


analyzer = SentimentIntensityAnalyzer()


def classify_sentiment(review):
  
    sentiment_score = analyzer.polarity_scores(review)['compound']
    
    # Classify the review based on the score
    if sentiment_score >= 0.05:
        return 'Positive'
    elif sentiment_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


df['sentiment'] = df['cleaned'].apply(classify_sentiment)


df[['cleaned', 'sentiment']].head()


# In[268]:


X = df['cleaned']
y = df['condition']
le = LabelEncoder()
y_encoded = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)







# In[269]:


pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])


param_grid = {
    # TF-IDF Hyperparameters
    'tfidf__max_features': [1000000],
    'tfidf__ngram_range': [ (1,1)],
    # Logistic Regression Hyperparameters
    'clf__C': [ 10],
}



grid_log= GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_log.fit(X_train, y_train)

y_pred_log= grid_log.predict(X_test)
y_train_pred_log=grid_log.predict(X_train)

print("Best Parameters:", grid_log.best_params_)
print("test Accuracy:", accuracy_score(y_test, y_pred_log))
print("training accuracy",accuracy_score(y_train,y_train_pred_log))

print("\nClassification Report:\n", classification_report(y_test, y_pred_log))


# In[270]:


def predict_condition(review):
    cleaned_review = clean_review_with_stop_and_stem(review)
    predicted_label = grid_log.predict([cleaned_review])[0]  # raw text goes directly here
    predicted_condition = le.inverse_transform([predicted_label])[0]
    return predicted_condition


new_review = "I've been feeling very down lately and have lost interest in activities I used to enjoy."
predicted_condition = predict_condition(new_review)
print(f"Predicted condition for the review: {predicted_condition}")


# In[271]:


df[df["Unnamed: 0"]==103458]


# In[272]:


review=df[df["Unnamed: 0"]==103458].review
print(f"Predicted condition for the review: {predict_condition(review)}")


# In[273]:


import pickle

with open("grid_log.pkl", "wb") as f:
    pickle.dump(grid_log, f)

with open("le.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Saved model.pkl and label_encoder.pkl")


# In[274]:


import dill
import pickle
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Load model and label encoder
with open("grid_log.pkl", "rb") as f:
     grid_log= pickle.load(f)

with open("le.pkl", "rb") as f:
    le = pickle.load(f)



def clean_review_with_stop_and_stem(text):
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    import re
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')


    # Setup
    stop = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove special characters and stopwords
    words = [word for word in text.split() if word not in stop]
    text_no_specials = re.sub(r'[^a-z0-9\s]', '', ' '.join(words))

    # Lemmatize and Stem each word
    processed_words = []
    for word in text_no_specials.split():
        lemma = lemmatizer.lemmatize(word)
        stemmed = stemmer.stem(lemma)
        processed_words.append(stemmed)

    return ' '.join(processed_words)


# Save both functions into one file using dill
functions = {
    "clean_review_with_stop_and_stem": clean_review_with_stop_and_stem,
    "predict_condition": predict_condition
}

with open("functions.pkl", "wb") as f:
    dill.dump(functions, f)

print("✅ Functions saved as functions.pkl")


# In[ ]:





# In[ ]:


import nbformat
from nbconvert import PythonExporter

# Path to the .ipynb file
notebook_path = 'drugmodel.ipynb'

# Create a PythonExporter
exporter = PythonExporter()

# Open and read the notebook
with open(notebook_path, 'r', encoding='utf-8') as notebook_file:
    notebook_content = notebook_file.read()

# Convert to Python code
python_code, _ = exporter.from_notebook_node(nbformat.reads(notebook_content, as_version=4))

# Save the Python code to a .py file
with open('drugmodel.py', 'w', encoding='utf-8') as python_file:
    python_file.write(python_code)

