from sklearn.naive_bayes import GaussianNB
import joblib
import pandas as pd

# Example: load your phishing dataset
data = pd.read_csv("phishing_dataset.csv")
X = data.drop("label", axis=1)
y = data["label"]

# Train the model
gnb = GaussianNB()
gnb.fit(X, y)

# Save the model
joblib.dump(gnb, "models/gnb_model.pkl")
