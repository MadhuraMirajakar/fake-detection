# train_improved_nlp.py
import re
import string
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import warnings
warnings.filterwarnings("ignore")

# Download NLTK resources (if not already)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# -------------------
# Text cleaning
# -------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Lemmatize and remove stopwords
    lemmatizer = WordNetLemmatizer()
    stops = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stops]
    return " ".join(tokens)

# -------------------
# Label mapping helpers
# -------------------
def to_binary_label(label_series):
    """
    Map LIAR-like labels to binary:
    fake = pants-fire, false, barely-true
    real = half-true, mostly-true, true
    Returns series of 0 (fake) / 1 (real)
    """
    fake = {'pants-fire', 'false', 'barely-true'}
    real = {'half-true', 'mostly-true', 'true'}
    # normalize strings
    s = label_series.astype(str).str.strip().str.lower()
    mapped = s.apply(lambda x: 0 if x in fake else (1 if x in real else np.nan))
    return mapped

# -------------------
# Training pipeline
# -------------------
def train_pipeline(train_path, valid_path, test_path, binary=True, n_iter_search=24, random_state=42):
    cols = ['id','label','statement','subject','speaker','job','state','party','c1','c2','c3','c4','c5','context']
    train_df = pd.read_csv(train_path, sep='\t', header=None, names=cols)
    val_df   = pd.read_csv(valid_path, sep='\t', header=None, names=cols)
    test_df  = pd.read_csv(test_path, sep='\t', header=None, names=cols)

    # drop missing statements
    for df in [train_df, val_df, test_df]:
        df.dropna(subset=['statement'], inplace=True)
        df['statement'] = df['statement'].apply(clean_text)
        df['context'] = df['context'].fillna('')
        # combine statement + context (helps short statements)
        df['text'] = (df['statement'] + " " + df['context']).str.strip()

    # Option: binary vs multiclass labels
    if binary:
        train_df['y'] = to_binary_label(train_df['label'])
        val_df['y']   = to_binary_label(val_df['label'])
        test_df['y']  = to_binary_label(test_df['label'])
        # Drop any rows where mapping failed
        train_df = train_df.dropna(subset=['y'])
        val_df   = val_df.dropna(subset=['y'])
        test_df  = test_df.dropna(subset=['y'])
        # convert to ints
        train_df['y'] = train_df['y'].astype(int)
        val_df['y']   = val_df['y'].astype(int)
        test_df['y']  = test_df['y'].astype(int)
    else:
        # multiclass: keep original labels (lowercased)
        train_df['y'] = train_df['label'].astype(str).str.strip().str.lower()
        val_df['y']   = val_df['label'].astype(str).str.strip().str.lower()
        test_df['y']  = test_df['label'].astype(str).str.strip().str.lower()

    # Merge train + val for training (we will evaluate on test)
    train_all = pd.concat([train_df, val_df], ignore_index=True)

    X_train = train_all['text'].values
    y_train = train_all['y'].values
    X_test  = test_df['text'].values
    y_test  = test_df['y'].values

    # Build TF-IDF feature union: word ngrams + char ngrams
    word_tfidf = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1,3),
        max_df=0.85,
        min_df=2,
        sublinear_tf=True,
        max_features=20000
    )
    char_tfidf = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3,6),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        max_features=10000
    )
    combined = FeatureUnion([("word", word_tfidf), ("char", char_tfidf)], n_jobs=-1)

    # SVD to reduce dimensionality (speeds up classifier, removes noise)
    svd = TruncatedSVD(n_components=300, random_state=random_state)

    # Use LogisticRegression with balanced class weights (works well)
    clf = LogisticRegression(solver='saga', max_iter=5000, class_weight='balanced', random_state=random_state)

    # Full pipeline
    pipe = Pipeline([
        ('tfidf', combined),
        ('svd', svd),
        ('clf', clf)
    ])

    # Parameter distribution for randomized search
    param_dist = {
        'svd__n_components': [150, 300, 400],
        'clf__C': [0.01, 0.1, 0.5, 1.0, 3.0, 5.0],
        'clf__penalty': ['l2', 'l1'],
    }

    # If multiclass, remove l1 penalty because saga + multiclass + l1 might be slow; still keeps options
    if not binary:
        param_dist['clf__penalty'] = ['l2']

    rs = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter_search,
                            cv=3, verbose=2, n_jobs=-1, random_state=random_state)

    print("🔎 Running RandomizedSearchCV (this may take a while)...")
    rs.fit(X_train, y_train)

    print("\n✅ Best params:")
    print(rs.best_params_)

    # Evaluate on the held-out test set
    y_pred = rs.predict(X_test)
    print("\n🧾 Test set classification report:")
    if binary:
        print(classification_report(y_test, y_pred, target_names=['fake','real']))
    else:
        print(classification_report(y_test, y_pred))

    # Save final pipeline
    fname = 'pipeline_binary.pkl' if binary else 'pipeline_multiclass.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(rs.best_estimator_, f)
    print(f"\n💾 Saved pipeline to: {fname}")

    return rs.best_estimator_

# -------------------
# Run training
# -------------------
if __name__ == "__main__":
    # change to False if you want to try multiclass (expect lower practical accuracy)
    # n_iter_search controls how extensive hyperparameter tuning is (more -> slower)
    model = train_pipeline('train.tsv', 'valid.tsv', 'test.tsv', binary=True, n_iter_search=32)
