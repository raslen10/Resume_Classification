# Resume Screening and Classification System


## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Technical Approach](#technical-approach)
- [Model Performance](#model-performance)


## Project Overview

This machine learning system automatically classifies resumes into 25 distinct job categories based on their content. The solution helps HR departments and recruitment agencies streamline their resume screening process.

## Key Features

- **Text Preprocessing Pipeline**
  - Special character removal
  - Lowercase conversion
  - Stopword removal
  - Tokenization

- **Machine Learning Models**
  - Random Forest Classifier
  - Logistic Regression
  - Multinomial Naive Bayes

- **Web Interface**
  - Flask-based web application
  - Simple resume submission form
  - Instant classification results

## Dataset

The dataset contains **962 labeled resumes** across 25 categories:

| Category                | Count |
|-------------------------|-------|
| Java Developer          | 84    |
| Testing                 | 70    |
| DevOps Engineer         | 55    |
| Python Developer        | 48    |
| Web Designing           | 45    |
| HR                      | 44    |
| ... (20 more categories)| ...   |

## Technical Approach

### Data Preprocessing
```python
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove special chars
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in word_tokenize(text) 
                   if word not in stopwords.words('english')])
    return text
```

## Feature Extraction

- **TF-IDF Vectorizer** with 1000 features  
- Text vectorization for machine learning input

## Models Evaluated

### Random Forest
- **Accuracy:** 98.96%  
- **Strengths:** Handles non-linear relationships well

### Logistic Regression (Selected Model)
- **Accuracy:** 99.48%  
- **Strengths:** Best overall performance

### Multinomial Naive Bayes
- **Accuracy:** 96.37%  
- **Strengths:** Fast training time

## Model Performance

### Accuracy Comparison

| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression  | 99.48%   |
| Random Forest        | 98.96%   |
| Multinomial NB       | 96.37%   |


