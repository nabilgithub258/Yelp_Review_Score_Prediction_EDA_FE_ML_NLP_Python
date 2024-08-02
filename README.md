
---

# Yelp Reviews Rating NLP Prediction

This project aims to predict user scores on Yelp reviews by building a machine learning model. The dataset contains user reviews and ratings ranging from 1 to 5.

## Key Features:

- **Data Preprocessing:**
  - **Handling Duplicates and Missing Values:** Ensured data quality by removing duplicates and addressing missing values.
- **Feature Engineering:**
  - **Rating Transformation:** Transformed the ratings column:
    - Scores 1 and 2 were combined into 0 (Poor).
    - Score 3 was converted to 1 (OK).
    - Scores 4 and 5 were combined into 2 (Good).
- **Text Vectorization:**
  - Used TF-IDF Vectorizer to convert review texts into numerical features.
- **Model Training:**
  - Developed a model to predict whether a user will give a good, OK, or poor review based on the text.
  - Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique) and `ImbPipeline`.

## Getting Started:
### Prerequisites
- Python 3.x
- Jupyter Notebook

## Results:
The project demonstrates effective handling of class imbalance and predicts user review scores based on text using advanced NLP techniques and machine learning models.

## Project Structure:
- **CSV/**: Contains the dataset files.
- **Code/**: Contains Jupyter notebooks and Python scripts for data preprocessing, EDA, and model training.
- **Plots/**: Contains plot files for dataset.
- **README.md**: Project documentation.

## Contributing:
Feel free to submit issues or pull requests if you have suggestions or improvements.

---
