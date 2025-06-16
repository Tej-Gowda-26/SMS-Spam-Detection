# SMS Spam Detection

This project implements an SMS Spam Detection system using a **Logistic Regression** model. It classifies SMS messages as either **Spam** or **Ham** (not spam) based on their textual content.

## Features

- Preprocessing of raw SMS text data
- TF-IDF vectorization (with bi-grams and stopword removal)
- Logistic Regression classifier
- Model evaluation with classification report
- Interactive UI built using **Streamlit**

## Technologies Used

- Python
- scikit-learn
- pandas
- NLTK
- Streamlit

## Dataset

This project uses the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) available on Kaggle. It consists of 5,574 SMS messages labeled as either ham or spam.

## How to Run Locally

1. **Clone the repository**

```bash
git clone https://github.com/Tej-gowda-26/sms-spam-detection.git
cd sms-spam-detection
```

2. **Create a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate
```   

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Download NLTK Resources**

```bash
python
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
```

5. **Train the model**

```bash
python src/train.py
```

6. **Evaluate the model**

```bash
python src/evaluate.py
```

6. **Predict the model**

```bash
python src/predict.py
```

7. **Run the Streamlit app**

```bash
streamlit run src/app.py
```

## Go to http://localhost:8501 in your browser to interact with the SMS Spam Detection interface.
