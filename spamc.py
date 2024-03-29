import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import string
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
nltk.download('stopwords')

#Read data from file csv
data=pd.read_csv('spam.csv')


def discribe_data(data):
    print("=>>data_sample:\n",data.head(5))
    '''
    Category                                            Message
    0      ham  Go until jurong point, crazy.. Available only ...
    1      ham                      Ok lar... Joking wif u oni...
    2     spam  Free entry in 2 a wkly comp to win FA Cup fina...
    3      ham  U dun say so early hor... U c already then say...
    4      ham  Nah I don't think he goes to usf, he lives aro...
    '''

    print("=>>data_shape: ",data.shape) #(5572, 2)

    #check missing data
    print("=>>check missing data: \n", data.isnull().sum())
    '''
    Category    0
    Message     0
    =>> no missing data
    '''
    #check imbalance data
    print("=>>check imbalance data: \n",data['Category'].value_counts())
    '''
    Category
    ham     4825
    spam     747
    Name: count, dtype: int64 '''
    plt.bar(['ham', 'spam'], data['Category'].value_counts())
    plt.show()
    #  imbalance data

# discribe_data(data)

def Text_Preprocessing(text):

    # Removing punctuation marks
    text = ''.join([char for char in text if char not in string.punctuation])

    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text into individual words
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
    words = tokenizer.tokenize(text)
    #words = TweetTokenizer(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]

    # Perform stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]

    # Join the stemmed words back into a single string
    processed_text = ' '.join(stemmed_words)

    return processed_text


def Data_Preprocessing(data):

    data['Message'] = data['Message'].apply(Text_Preprocessing)

    #drop duplicates keep first
    data.drop_duplicates(keep='first', inplace=True)

    #covert ham into 0 and spam into 1
    data['Category']=data['Category'].map({'ham':0,'spam':1})

    X = data['Message']
    y = data['Category']    

    # 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

    X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the training data and transform the training data
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)  
    X_validation = vectorizer.transform(X_validation)   

    return X_train, X_test, X_validation, y_train, y_test, y_validation

X_train, X_test, X_validation, y_train, y_test, y_validation = Data_Preprocessing(data)

def find_model(X_train, y_train, X_validation, y_validation ):

    bc = BaggingClassifier(n_estimators=50, random_state=42)
    etc = ExtraTreesClassifier(n_estimators=50, random_state=42)
    gbdt = GradientBoostingClassifier(n_estimators=50, random_state=42)
    xgb = XGBClassifier(n_estimators=50, random_state=42)
    svc = SVC(kernel="sigmoid", gamma=1.0)
    mnb = MultinomialNB()
    dtc = DecisionTreeClassifier(max_depth=10)
    lrc = LogisticRegression(solver='liblinear', penalty='l1')
    rfc = RandomForestClassifier(n_estimators=50, random_state=42)
    abc = AdaBoostClassifier(n_estimators=50, random_state=42)

    models = {
    'BaggingClassifier': bc,
    'ExtraTreesClassifier': etc,
    'GradientBoostingClassifier': gbdt,
    'XGBClassifier': xgb,
    'SVC': svc,
    'MultinomialNB': mnb,
    'DecisionTreeClassifier': dtc,
    'LogisticRegression': lrc,
    'RandomForestClassifier': rfc,
    'Adaboost': abc
    }   

    def train_classifier(model, X_train, y_train, X_validation, y_validation):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_validation)

        accuracy = accuracy_score( y_validation, y_pred)
        precision = precision_score( y_validation, y_pred)
        recall = recall_score( y_validation, y_pred)
        f1 = f1_score( y_validation, y_pred)

        return accuracy, precision , recall, f1
    
    accuracy_scores = []
    precision_scores = []

    results = {
    'Algorithm': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1': []
    }

    for name, model in models.items():
        current_accuracy, current_precision, current_recall, current_f1 = train_classifier(model, X_train, y_train, X_validation, y_validation)
        results['Algorithm'].append(name)
        results['Accuracy'].append(round(current_accuracy, 2))
        results['Precision'].append(round(current_precision, 2))
        results['Recall'].append(round(current_recall, 2))
        results['F1'].append(round(current_f1, 2))

    # Tạo DataFrame từ dictionary
    results_df = pd.DataFrame(results)

    # In bảng kết quả
    print(results_df)


find_model(X_train, y_train, X_validation, y_validation )