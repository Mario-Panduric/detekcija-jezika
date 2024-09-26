import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import time

def showResult(model):
    model.fit(X_train, y_train)
    output = model.predict(cv.transform([text]).toarray())
    st.write('Result: ' + str(output))
    labels = ['Dutch', 'English', 'Estonian', 'French', 'Indonesian', 'Latin', 'Portugese', 'Romanian', 'Spanish', 'Swedish', 'Turskish']
    st.text("Confusion matrix:")
    cmatrix = confusion_matrix(y_test, model.predict(X_test))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cmatrix, display_labels=labels)
    fig, ax = plt.subplots()
    cm_display.plot(ax=ax, cmap=plt.cm.Blues)
    plt.xticks(rotation=90)
    st.pyplot(fig)
    st.write('Classification report:')
    clreport = classification_report(y_test, model.predict(X_test))
    st.text(clreport)
 

    
    

st.title('Language detection')

c1, c2, c3, c4, c5 = st.columns(5)

compare = st.button('Compare')
with c1:
    nb = st.button ('Naive Bayes') 

with c2:
    svm = st.button ('Support Vector Machines')

with c3:
    knn = st.button ('K Nearest Neighbors')

with c4:
    rf = st.button('Random Forest')

with c5:
    lreg  = st.button('Logistic Regression')

text = st.text_input('Text for language detection')

data = pd.read_csv(r"dataset.csv")
data = pd.DataFrame(data)

data = data[data["language"].str.contains("Chinese") == False]
data = data[data["language"].str.contains("Thai") == False]
data = data[data["language"].str.contains("Persian") == False]
data = data[data["language"].str.contains("Japanese") == False]
data = data[data["language"].str.contains("Hindi") == False]
data = data[data["language"].str.contains("Pushto") == False]
data = data[data["language"].str.contains("Arabic") == False]
data = data[data["language"].str.contains("Tamil") == False]
data = data[data["language"].str.contains("Urdu") == False]
data = data[data["language"].str.contains("Russian") == False]
data = data[data["language"].str.contains("Korean") == False]



x = np.array(data["Text"])

y = np.array(data["language"])


cv = CountVectorizer()

X = cv.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=1)

global model 

if(nb):
    model = MultinomialNB()
    showResult(model)
    
if(svm):
    model = LinearSVC()
    showResult(model)
    

if(knn):
    model = KNeighborsClassifier(n_neighbors=45, weights="distance")
    showResult(model)
   

if(rf):
     model = RandomForestClassifier(n_estimators=11, random_state=1, n_jobs=4)
     showResult(model)

if(lreg):
    model = LogisticRegression(multi_class='multinomial', solver='sag')
    showResult(model)

if(compare):
    model_b = MultinomialNB()
    model_svm = LinearSVC()
    model_rf = RandomForestClassifier(n_estimators=11, random_state=1, n_jobs=4)
    model_knn = KNeighborsClassifier(n_neighbors=11, weights="distance")
    model_lreg = LogisticRegression(multi_class='multinomial', solver='sag')
    model_b.fit(X_train, y_train)
    model_svm.fit(X_train, y_train)
    model_knn.fit(X_train, y_train)
    model_lreg.fit(X_train, y_train)
    model_rf.fit(X_train, y_train)
    models = [model_b, model_svm, model_rf, model_knn, model_lreg]
    model_names = ['MultinomialNB', 'LinearSVC', 'RandomForest', 'KNN', 'LogisticRegression']

    metrics = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'Execution Time': []
    }

    for model in models:
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        execution_time = time.time() - start_time
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        metrics['Accuracy'].append(accuracy)
        metrics['Precision'].append(precision)
        metrics['Recall'].append(recall)
        metrics['F1 Score'].append(f1)
        metrics['Execution Time'].append(execution_time)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    x = np.arange(len(model_names))
    width = 0.2

    ax1.bar(x - width, metrics['Accuracy'], width, label='Accuracy')
    ax1.bar(x, metrics['Precision'], width, label='Precision')
    ax1.bar(x + width, metrics['Recall'], width, label='Recall')
    ax1.bar(x + (2 * width), metrics['F1 Score'], width, label='F1 Score')

    ax2.plot(x, metrics['Execution Time'], color='blue', marker='o', label='Execution Time')

    ax1.set_xlabel('Algorithms')
    ax1.set_ylabel('Values')
    ax2.set_ylabel('Execution Time (s)')
    ax1.set_title('Metrics of algorithms')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    st.pyplot(fig)

    

