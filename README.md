# Resume Classification Project

## Project Overview
This project focuses on classifying resumes into different job categories using machine learning techniques. The goal is to automate the process of categorizing resumes based on their content, which can be highly beneficial for HR departments and recruitment agencies.

## Dataset
- **Source**: UpdatedResumeDataSet.csv
- **Size**: 962 resumes
- **Features**:
  - Category (25 unique job categories)
  - Resume text content
- **Sample Categories**: Data Science, Testing, Java Developer, DevOps Engineer, etc.

## Technical Approach

### 1. Data Exploration
- Examined dataset structure and distribution of categories
- Visualized category distribution using matplotlib
- Analyzed text content using word frequency and word clouds

### 2. Text Preprocessing
- Implemented cleaning functions to:
  - Remove special characters
  - Convert to lowercase
  - Remove stopwords
  - Tokenize text
- Applied CountVectorizer and TF-IDF for text vectorization

### 3. Model Implementation
Tested three classification algorithms:
1. **Random Forest Classifier**
2. **Logistic Regression**
3. **Multinomial Naive Bayes**

### 4. Evaluation Metrics
- Accuracy score
- Classification report (precision, recall, f1-score)

## Key Results
- Achieved classification accuracy of [X]% (to be filled with actual results)
- [Best performing model] demonstrated strongest performance
- Most discriminative features identified through EDA

## Code Structure
The project is implemented in a Jupyter notebook (`Resume_Classification.ipynb`) with the following sections:

1. **Import Libraries**
   - pandas, numpy for data manipulation
   - sklearn for machine learning
   - nltk for text processing
   - matplotlib for visualization

2. **Data Loading & Exploration**
   - Shape and info of dataset
   - Missing value analysis
   - Category distribution visualization

3. **Text Analysis**
   - Word frequency analysis
   - Text cleaning pipeline
   - Feature extraction

4. **Model Training**
   - Data splitting (train/test)
   - Model implementation
   - Performance evaluation

## How to Run
1. Ensure required libraries are installed
2. Download the dataset
3. Run the Jupyter notebook cells sequentially

## Future Improvements
- Experiment with more advanced NLP techniques (word embeddings)
- Try deep learning approaches
- Expand dataset for better generalization
- Develop a web interface for practical use

## Dependencies
- Python 3.x
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- wordcloud

