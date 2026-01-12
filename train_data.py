import pandas as pd
import joblib
from backend.nlp_process import preprocess_text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report


# 1. Load dataset

df = pd.read_csv('/Users/AI_MEETING_SUMMARIZER/data/train_dataset.csv')

df['clean_text'] = df['Text'].apply(preprocess_text)

x = df['clean_text']
y = df['label']

# Convert tokens → string
x_str = [' '.join(tokens) for tokens in x]


# 2. Train-test split

x_train, x_test, y_train, y_test = train_test_split(
    x_str,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 3. Pipeline (TF-IDF + SGD)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        strip_accents='unicode'
    )),
    ('clf', SGDClassifier(
        max_iter=5000,
        random_state=42,
        class_weight='balanced'
    ))
])


# 4. GridSearchCV

param_grid = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'tfidf__min_df': [2, 5],
    'tfidf__max_features': [30000, 50000],
    'clf__loss': ['hinge'],
    'clf__alpha': [1e-4, 5e-4, 1e-3]
}

model = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)


# 5. Train

model.fit(x_train, y_train)


# 6. Evaluate

y_pred = model.predict(x_test)

print("Best parameters:")
print(model.best_params_)

print("\nClassification report:\n")
print(classification_report(y_test, y_pred))

#save file 

joblib.dump(model.best_estimator_, "whisper_meeting_classifier.pkl")

print("\n✅ Model saved as whisper_meeting_classifier.pkl")
