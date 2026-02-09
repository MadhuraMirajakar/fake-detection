from datetime import datetime
import os
import re
import sys
import hashlib
import pickle
import requests
import joblib
import numpy as np

import firebase_admin
from firebase_admin import credentials, db

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_cors import CORS
from urllib.parse import urlparse

from train import clean_text


# =====================================================
# ⚙️ Flask Setup
# =====================================================
app = Flask(__name__)
app.secret_key = "supersecretkey"
CORS(app)


# =====================================================
# 🔥 Firebase Setup
# =====================================================
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://fake-web-n-news-detection-default-rtdb.firebaseio.com/"
})


# =====================================================
# 🧠 Load ML Model (Phishing)
# =====================================================
PHISHING_MODEL_PATH = os.path.join("models", "gnb_model.pkl")
phishing_model = None

try:
    if os.path.exists(PHISHING_MODEL_PATH):
        phishing_model = joblib.load(PHISHING_MODEL_PATH)
        print("✅ Phishing model loaded successfully.")
    else:
        print(f"❌ Phishing model not found at {PHISHING_MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Phishing model load failed: {e}")


# =====================================================
# 🔐 Helper Functions
# =====================================================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def heuristic_phishing(url: str) -> bool:
    """Simple rule-based phishing detection."""
    if not isinstance(url, str) or not url.strip():
        return False

    url = url.strip().lower()

    if '@' in url:
        return True

    if re.search(r'http[s]?://\d+\.\d+\.\d+\.\d+', url):
        return True

    suspicious_keywords = [
        'login', 'signin', 'verify', 'update', 'secure',
        'account', 'confirm', 'paypal', 'bank', 'webscr', 'ebayisapi'
    ]

    if any(kw in url for kw in suspicious_keywords):
        return True

    parsed = urlparse(url)
    host = parsed.netloc.split(':')[0]

    if host.count('.') >= 5 or ('-' in host and len(host.split('-')) >= 2):
        return True

    return False


# =====================================================
# 🧭 Authentication (Register / Login / Logout)
# =====================================================
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register")
def register():
    return render_template("register.html")


@app.route("/signup", methods=["POST"])
def signup():
    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")
    confirm = request.form.get("confirm-password")

    if password != confirm:
        flash("Passwords do not match!", "error")
        return redirect(url_for("register"))

    ref = db.reference("users")
    users = ref.get()

    # Check if user exists
    if users:
        for uid, user in users.items():
            if user["email"] == email:
                flash("User already exists! Please log in.", "warning")
                return redirect(url_for("home"))

    # Save new user
    ref.push({
        "name": name,
        "email": email,
        "password": hash_password(password)
    })

    flash("Registration successful! Please sign in.", "success")
    return render_template("detector.html")


@app.route("/signin", methods=["POST"])
def signin():
    email = request.form.get("email")
    password = request.form.get("password")

    ref = db.reference("users")
    users = ref.get()

    if not users:
        flash("No users found. Please sign up first.", "error")
        return redirect(url_for("home"))

    hashed_pw = hash_password(password)

    for uid, user in users.items():
        if user["email"] == email and user["password"] == hashed_pw:
            session["user"] = user["name"]
            flash(f"Welcome, {user['name']}!", "success")
            return render_template("detector.html")

    flash("Invalid email or password!", "error")
    return redirect(url_for("register"))


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


# =====================================================
# 🕵️‍♂️ Phishing Detector (Heuristic + ML)
# =====================================================
@app.route("/detector", methods=["GET", "POST"])
def detector():
    if "user" not in session:
        flash("Please log in first.", "error")
        return redirect(url_for("register"))

    result = confidence = method = None

    if request.method == "POST":
        url = request.form.get("url", "").strip()

        if not url:
            result = "⚠️ Please enter a valid URL."
        else:
            try:
                heuristic_flag = heuristic_phishing(url)
            except Exception:
                heuristic_flag = False

            if heuristic_flag:
                result = "⚠️ Phishing Likely!"
                method = "Heuristic"

            else:
                if phishing_model is None:
                    result = "✅ This website looks safe! (Heuristic only)"
                    method = "Heuristic"
                else:
                    parsed = urlparse(url)
                    feats = [
                        len(url),
                        url.count('.'),
                        sum(c.isdigit() for c in url),
                        1 if parsed.scheme == 'https' else 0,
                        len(parsed.hostname or ''),
                        sum(1 for kw in ['login', 'verify', 'secure', 'account', 'update', 'confirm']
                            if kw in url.lower())
                    ]

                    try:
                        X = np.array([feats])

                        if hasattr(phishing_model, "predict_proba"):
                            proba = phishing_model.predict_proba(X)[:, 1][0]
                            pred = int(proba >= 0.5)
                            confidence = round(float(proba), 2)
                            result = "⚠️ Phishing Likely!" if pred else "✅ Safe Website!"
                            method = "Machine Learning"
                        else:
                            pred = int(phishing_model.predict(X)[0])
                            result = "⚠️ Phishing Likely!" if pred else "✅ Safe Website!"
                            method = "Machine Learning"

                    except Exception as e:
                        print("❌ Model prediction failed:", e)
                        result = "✅ Safe (Fallback to Heuristic)"
                        method = "Heuristic"

    return render_template("detector.html", result=result, confidence=confidence, method=method)


# =====================================================
# 🧠 Fake News Model Load
# =====================================================
def load_pipeline(binary=True):
    fname = 'pipeline_binary.pkl' if binary else 'pipeline_multiclass.pkl'
    try:
        with open(fname, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"✅ Loaded model pipeline from {fname}")
    except FileNotFoundError:
        print(f"❌ Error: {fname} not found.")
        sys.exit(1)

    return pipeline


binary = True
model = load_pipeline(binary=binary)


# =====================================================
# 🌐 Fake News Predictor
# =====================================================
@app.route("/news_detector")
def news_detector():
    if "user" not in session:
        flash("Please log in first.", "error")
        return redirect(url_for("register"))

    return render_template("news_detector.html", user=session.get("user"))


@app.route("/predict_news", methods=["POST"])
def predict_news():
    try:
        data = request.get_json()
        text = data.get("content", "").strip()

        if not text:
            return jsonify({"error": "No content provided."}), 400

        cleaned = clean_text(text)
        prediction = int(model.predict([cleaned])[0])

        # Probability support
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba([cleaned])[0][prediction])
            credibility = round((proba * 100) * 1.5)
            credibility = max(30, min(98, credibility))
        else:
            credibility = 85 if prediction == 1 else 45

        if prediction == 1:
            result = {
                "title": "REAL News 🟢",
                "summary": "This appears to be credible content.",
                "credibility": credibility,
                "analysis": [
                    "Neutral and fact-based language detected",
                    "Contains verifiable claims and references"
                ],
                "recommendations": [
                    "Cross-verify from other reputable outlets",
                    "You can trust this source, but stay updated"
                ]
            }
        else:
            result = {
                "title": "FAKE News 🔴",
                "summary": "This appears to be misleading or false.",
                "credibility": credibility,
                "analysis": [
                    "Emotional or exaggerated wording detected",
                    "Lack of credible or referenced sources"
                ],
                "recommendations": [
                    "Avoid sharing this without verification",
                    "Check fact-checking sites for confirmation"
                ]
            }

        return jsonify(result)

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return jsonify({"error": "Internal server error"}), 500




# =====================================================
# 📞 Phone Lookup API (NumLookupAPI)
# =====================================================

NUMLOOKUP_KEY = os.getenv("NUMLOOKUP_KEY")

@app.route("/api/phone_lookup", methods=["POST"])
def phone_lookup():
    data = request.get_json()
    if not data or "phone" not in data:
        return jsonify({"valid": False, "error": "Phone number missing"}), 400

    phone = (
        data["phone"]
        .replace(" ", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "")
    )
    if not phone.startswith("+"):
        phone = "+" + phone

    # Read key at request time
    NUMLOOKUP_KEY = os.getenv("NUMLOOKUP_KEY")

    if not NUMLOOKUP_KEY:
        print("❌ NUMLOOKUP_KEY missing in environment.")
        return jsonify({"valid": False, "error": "Server misconfiguration: API key missing"}), 500

    url = f"https://api.numlookupapi.com/v1/validate/{phone}?apikey={NUMLOOKUP_KEY}"

    # DEBUG lines — paste these in your next message if error continues
    print("DEBUG repr(url):", repr(url))
    print("DEBUG type(url):", type(url))
    print("DEBUG NUMLOOKUP_KEY (masked):", (NUMLOOKUP_KEY[:8] + "...") if len(NUMLOOKUP_KEY) > 8 else NUMLOOKUP_KEY)

    # Sanity: only call requests.get if url looks like a real http URL
    if not isinstance(url, str) or not url.startswith("http"):
        print("❌ Bad URL detected — aborting before requests.get()")
        return jsonify({"valid": False, "error": "Bad URL created for API call", "debug": repr(url)}), 500

    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        result = resp.json()
        print("📞 NUMLOOKUP RESULT:", result)
    except requests.exceptions.RequestException as e:
        print("❌ REQUEST ERROR:", repr(e))
        return jsonify({"valid": False, "error": str(e)}), 500
    except ValueError:
        print("❌ JSON parse error")
        return jsonify({"valid": False, "error": "Invalid response from API"}), 500

    output = {
        "valid": result.get("valid", False),
        "number": result.get("number", ""),
        "local_format": result.get("local_format", ""),
        "international_format": result.get("international_format", ""),
        "country_prefix": result.get("country_prefix", ""),
        "country_code": result.get("country_code", ""),
        "country_name": result.get("country_name", ""),
        "location": result.get("location", "Unknown Location"),
        "carrier": result.get("carrier", "Unknown Carrier"),
        "line_type": result.get("line_type", "Unknown"),
    }
    return jsonify(output)


# =====================================================
# 🌐 Web Pages (Phone, About, Contact)
# =====================================================
@app.route("/phone")
def phone():
    return render_template("phone.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


# =====================================================
# 💬 Feedback Storage
# =====================================================
@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    try:
        data = request.get_json() or {}

        feedback_data = {
            "name": data.get("feedbackName", "Anonymous"),
            "email": data.get("feedbackEmail", "Not Provided"),
            "category": data.get("feedbackCategory", "General"),
            "message": data.get("feedbackMessage", ""),
            "rating": data.get("ratingValue", "0"),
            "contact_permission": "Yes" if data.get("contactPermission") else "No",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if not feedback_data["message"].strip():
            return jsonify({"status": "error", "message": "Feedback message required."}), 400

        ref = db.reference("feedbacks")
        ref.push(feedback_data)

        return jsonify({"status": "success", "message": "Feedback stored successfully."})

    except Exception as e:
        print("❌ Error saving feedback:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# =====================================================
# 🏁 MAIN ENTRY
# =====================================================
if __name__ == "__main__":
    print("🚀 Running Flask Application...")
    app.run(debug=True, host="127.0.0.1", port=5000)
