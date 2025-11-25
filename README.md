# 📚 Book Review Sentiment Classifier

Binary text classifier that predicts whether a book review is **positive** or **negative** using TF-IDF features and compares a **logistic regression baseline** to a **neural network**.

This project was completed as part of the **Break Through Tech AI – ML Foundations** course and is one of my first end-to-end NLP pipelines. I framed the problem from a **business & product** angle: how would a company like Amazon use this model to improve the customer experience and make better decisions?

---

## 🎯 Project Overview

**Goal**  
Given a book review (free-text), predict whether the review is **positive** (`True`) or **negative** (`False`).

**Why it matters (business context)**

A sentiment model like this can help:

- Surface helpful products by aggregating sentiment at the book level  
- Prioritize customer support on highly negative reviews  
- Support **KPI-driven content strategy** – which themes/features drive positive sentiment?  
- Reduce manual review time by automatically flagging problematic feedback  

This is a classic example of how **data science bridges customer behavior, product decisions, and business outcomes**—very aligned with where I want to grow at the intersection of **data, marketing, and product**.

---

## 🧾 Dataset

- Source: **Book Reviews dataset** (provided in course materials)  
- Size: **1,973** reviews  
- Columns:
  - `Review` – free-text book review  
  - `Positive Review` – boolean label (`True` = positive, `False` = negative)

The full dataset used in this project is stored at:

data/bookReviewsData.csv

The class distribution is roughly balanced:

- ~50% positive  
- ~50% negative  

so no heavy class rebalancing was necessary.

---

## 🧹 Data Preparation & Feature Engineering

Key steps in the notebook:

### 1. Exploratory Data Analysis (EDA)

- Checked class balance with `value_counts()` and a `seaborn.countplot`
- Engineered:
  - `review_length` – number of characters  
  - `word_count` – number of words  

**Observed:**

- `review_length` and `word_count` are highly correlated (~0.997), so they carry almost the same information.

---

### 2. Text Vectorization (TF-IDF)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)
X = tfidf.fit_transform(df["Review"]).toarray()
y = df["Positive Review"].astype(int)

---

### 3. Train / Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

I experimented with the length-based features, but since they were so strongly correlated and the model already gets rich signal from TF-IDF, I kept the final modeling focused on **text features**.

---

## 🤖 Models

I compared a **neural network** to a **logistic regression baseline**.

---

### 1️⃣ Neural Network (Keras)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

nn_model = Sequential()
nn_model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
nn_model.add(Dropout(0.3))
nn_model.add(Dense(32, activation="relu"))
nn_model.add(Dropout(0.3))
nn_model.add(Dense(1, activation="sigmoid"))

nn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

To reduce overfitting, I used:

- `Dropout`  
- `EarlyStopping`  
- `ReduceLROnPlateau`  

Even with regularization, the neural network:

- Reached **100% training accuracy** quickly  
- Plateaued around **~0.79–0.80 validation accuracy**  
- Achieved **~0.76 test accuracy**  

This is a classic overfitting pattern: the network is starting to **memorize** the training data rather than generalize.

---

### 2️⃣ Logistic Regression (Baseline)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Test Accuracy: {lr_accuracy:.4f}")

**Result**

- ✅ Logistic Regression Test Accuracy: ≈ **0.8152**  
- ❌ Neural Network Test Accuracy: ≈ **0.76**

In this setup:

> A simple, well-tuned linear model (**logistic regression**) outperformed the neural network on unseen data.

---

## 📊 Key Results & Takeaways

- **Best test accuracy**: ~**81.5%** (logistic regression)

**Neural network:**

- Perfect training accuracy  
- Lower test accuracy and clear signs of overfitting  

**What I learned**

- More complex ≠ always better  
- Always compare against a **simple baseline**  
- Validation/test metrics > “coolness” of the model architecture  
- Overfitting patterns are easy to spot when plotting training vs. validation loss/accuracy  

From a real-world perspective, I would deploy the **logistic regression** model here: it’s simpler, more interpretable, and generalizes better.

---

## 🗂 Project Structure

```text
book-review-sentiment-classifier/
├── data/
│   └── bookReviewsData.csv                # Book review text + sentiment label
├── notebook/
│   └── book_review_sentiment_classifier.ipynb  # Full EDA & modeling
├── README.md                              # Project overview (this file)
└── requirements.txt                       # (optional) Python dependencies

## ▶️ How to Run

1. **Clone the repository**

```bash
git clone https://github.com/nancy1404/book-review-sentiment-classifier.git
cd book-review-sentiment-classifier

2. **(Optional) Create & activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

3. **Install dependencies (if requirements.txt is present)**

```bash
pip install -r requirements.txt

4. **Open the notebook**

```bash
jupyter notebook notebook/book_review_sentiment_classifier.ipynb

Run the cells in order to reproduce the analysis, modeling, and evaluation.

---

## 🚀 Possible Next Steps

If I extend this project, I’d like to:

- Try **n-gram TF-IDF** (including bigrams) and compare performance  
- Evaluate additional models:
  - Linear SVM (`LinearSVC`)
  - Regularized logistic regression with different `C` values  
- Add **model explainability**:
  - Display top positive/negative words contributing to predictions  
- Build a small **demo app or API**, e.g.:
  - Simple UI where users paste a review and see the predicted sentiment  

---

## 🙋‍♀️ About Me

I’m **Nancy (Nakyung) Kwak**, a Statistics & Data Science major at the **University of Texas at Austin** and a **Break Through Tech AI Fellow** at Cornell Tech.

I’m especially interested in:

- Sports & marketing analytics  
- ML applications in customer experience and content strategy  
- Using data to bridge technical insights and real business decisions  

You can also find me here:

- GitHub: [@nancy1404](https://github.com/nancy1404)  
- LinkedIn: [nakyungnancy](https://www.linkedin.com/in/nakyungnancy)

