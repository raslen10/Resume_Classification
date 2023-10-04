from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Charger le jeu de données
data = pd.read_csv('UpdatedResumeDataSet.csv')

# Fonction pour nettoyer le texte
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer les caractères spéciaux
    text = text.lower()  # Mettre en minuscules
    text = ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english')])  # Supprimer les stopwords
    return text

# Appliquer la fonction de nettoyage à la colonne 'Resume'
data['Cleaned_Resume'] = data['Resume'].apply(clean_text)

# Séparer les données en caractéristiques (X) et cible (y)
X = data['Cleaned_Resume']
y = data['Category']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le vecteur TF-IDF en dehors de la fonction predict
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

# Adapter et transformer les données texte pour l'entraînement
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transformer les données de test
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Charger le modèle pickle pour la classification de texte
model_filename = "resume_classification.pkl"
with open(model_filename, 'rb') as model_file:
    text_model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    resume = [request.form['resume']]

    # Transformez le texte en utilisant le vecteur TF-IDF
    resume_tfidf = tfidf_vectorizer.transform(resume)

    # Utilisez le modèle pickle pour prédire la catégorie
    prediction = text_model.predict(resume_tfidf)

    # Remplacez les codes de catégorie par le nom de la catégorie correspondante si nécessaire
    categories = {
        0: 'Advocate', 1: 'Arts', 2: 'Automation Testing', 3: 'Blockchain',
        4: 'Business Analyst', 5: 'Civil Engineer', 6: 'Data Science', 7: 'Database',
        8: 'DevOps Engineer', 9: 'DotNet Developer', 10: 'ETL Developer', 11: 'Electrical Engineering',
        12: 'HR', 13: 'Hadoop', 14: 'Health and fitness', 15: 'Java Developer', 16: 'Mechanical Engineer',
        17: 'Network Security Engineer', 18: 'Operations Manager', 19: 'PMO', 20: 'Python Developer',
        21: 'SAP Developer', 22: 'Sales', 23: 'Testing', 24: 'Web Designing'
    }

    predicted_category = categories.get(prediction[0], 'Inconnue')

    return render_template('index.html', prediction=predicted_category)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
