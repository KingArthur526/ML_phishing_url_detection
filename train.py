"""
train.py — Phishing URL Detector
Central Algorithm: URL Feature Engineering + Random Forest Classifier
"""

import re
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import urllib.parse
import tldextract
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

logging.getLogger("tldextract").setLevel(logging.CRITICAL)

# ── Suspicious keywords common in phishing URLs ────────────────────────────
PHISHING_KEYWORDS = [
    "login", "signin", "verify", "update", "secure", "account",
    "banking", "confirm", "password", "credential", "ebay", "paypal",
    "amazon", "apple", "microsoft", "google", "facebook", "support",
    "webscr", "cmd", "dispatch", "click", "free", "lucky", "winner",
    "urgent", "alert", "suspended", "unusual", "activity",
]

SHORTENERS = [
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "rebrand.ly", "shorte.st", "cutt.ly",
]

# ── CENTRAL ALGORITHM STEP 1: Feature Engineering ─────────────────────────
def extract_features(url: str) -> list:
    """
    Extracts 20 numeric features from a raw URL string.
    No network calls — everything derived from the URL itself.
    """
    try:
        url = str(url).strip().encode('ascii', errors='ignore').decode('ascii')
        if not url:
            return [0] * 20
        parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
        ext = tldextract.extract(url)
        hostname = parsed.netloc or ""
        path = parsed.path or ""
        query = parsed.query or ""
        full = url.lower()
        return [
            len(url),                                               # 1. URL length
            len(hostname),                                          # 2. Hostname length
            len(path),                                              # 3. Path length
            len(query),                                             # 4. Query string length
            url.count("."),                                         # 5. Dot count
            url.count("-"),                                         # 6. Hyphen count
            url.count("@"),                                         # 7. @ symbol
            url.count("//"),                                        # 8. Double slash
            url.count("?"),                                         # 9. Question marks
            url.count("="),                                         # 10. Equals signs
            url.count("%"),                                         # 11. Percent-encoded chars
            sum(c.isdigit() for c in hostname),                     # 12. Digits in hostname
            1 if parsed.scheme == "https" else 0,                   # 13. Uses HTTPS
            len(ext.subdomain.split(".")) if ext.subdomain else 0,  # 14. Subdomain depth
            hostname.count("-"),                                    # 15. Hyphens in hostname
            1 if re.search(r"\d{1,3}(\.\d{1,3}){3}", hostname) else 0,  # 16. IP as host
            sum(1 for kw in PHISHING_KEYWORDS if kw in full),      # 17. Phishing keywords
            1 if any(s in full for s in SHORTENERS) else 0,        # 18. URL shortener
            1 if len(ext.domain) <= 3 else 0,                      # 19. Short domain
            len(ext.suffix.split(".")) if ext.suffix else 0,       # 20. TLD depth
        ]
    except Exception:
        return [0] * 20


FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "query_length",
    "dot_count", "hyphen_count", "at_symbol", "double_slash",
    "question_marks", "equals_signs", "percent_encoded", "digits_in_hostname",
    "uses_https", "subdomain_depth", "hyphens_in_hostname", "ip_as_hostname",
    "phishing_keyword_count", "uses_shortener", "short_domain", "tld_depth",
]

# ── Synthetic fallback data (used when no CSV is provided) ────────────────
LEGIT_URLS = [
    "https://www.google.com/search?q=weather", "https://github.com/numpy/numpy",
    "https://stackoverflow.com/questions/12345", "https://en.wikipedia.org/wiki/Machine_learning",
    "https://www.amazon.com/dp/B08N5WRWNW", "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://mail.google.com/mail/u/0/#inbox", "https://docs.python.org/3/library/os.html",
    "https://www.bbc.com/news/technology", "https://www.nytimes.com/section/technology",
    "https://www.reddit.com/r/MachineLearning", "https://twitter.com/home",
    "https://www.linkedin.com/in/profile", "https://www.apple.com/iphone",
    "https://www.microsoft.com/en-us/windows", "https://www.netflix.com/browse",
    "https://www.spotify.com/account/overview", "https://www.dropbox.com/home",
    "https://www.paypal.com/myaccount/summary", "https://www.ebay.com/itm/12345678",
]

PHISHING_URLS = [
    "http://paypal-security-update.com/login?cmd=verify&account=12345",
    "http://192.168.1.1/banking/login.php?redirect=paypal.com",
    "http://secure-amazon-account-verify.xyz/signin",
    "http://apple-id-confirm.ru/update/password",
    "http://microsoft-support-alert.com/urgent/credential",
    "http://ebay-account-suspended.net/verify?user=confirm",
    "http://bit.ly/3xR9qW2", "http://login-facebook-account.tk/signin.php",
    "http://google-account-security-alert.com/verify",
    "http://netflix-billing-update.gq/account/payment",
    "http://amazon.com.account-verify.ru/login",
    "http://paypal.com.secure-login.xyz/webscr?cmd=dispatch",
    "http://chase-bank-secure.com/account/login?verify=true",
    "http://wellsfargo-alert.com/signin?suspended=true",
    "http://bankofamerica-verify.net/secure/login",
    "http://update-your-apple-id.com/account/confirm",
    "http://free-iphone-winner.click/claim?user=lucky",
    "http://urgent-paypal-activity.com/account/unusual",
    "http://secure.paypa1.com/login", "http://arnazon.com/dp/signin",
]


def train_model(save_path="phishing_model.pkl", csv_path=None, samples_per_class=15000):
    if csv_path:
        print(f"Loading real dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Total: {len(df):,} | {df['type'].value_counts().to_dict()}\n")

        parts = [df[df['type'] == c].sample(min(samples_per_class, (df['type'] == c).sum()), random_state=42)
                 for c in df['type'].unique()]
        sample = pd.concat(parts).reset_index(drop=True)
        sample['label'] = (sample['type'] == 'phishing').astype(int)
        urls = sample['url'].astype(str).tolist()
        labels = sample['label'].tolist()
        print(f"Training on {len(urls):,} real URLs...\n")
    else:
        print("No CSV provided — using synthetic training data...")
        urls = LEGIT_URLS * 3 + PHISHING_URLS * 3
        labels = [0] * (len(LEGIT_URLS) * 3) + [1] * (len(PHISHING_URLS) * 3)

    print("Extracting features...")
    X = np.array([extract_features(u) for u in urls])
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}\n")
    print("Training Random Forest...")

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\n{'='*50}")
    print(f"✅ Test Accuracy: {acc*100:.2f}%")
    print(f"{'='*50}")
    print(classification_report(y_test, preds, target_names=["Not Phishing", "Phishing"]))

    with open(save_path, "wb") as f:
        pickle.dump({"model": clf, "feature_names": FEATURE_NAMES, "extract_features": extract_features}, f)

    print(f"✅ Model saved to {save_path}")
    return clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to malicious_phish.csv")
    parser.add_argument("--samples", type=int, default=15000, help="Samples per class")
    args = parser.parse_args()
    train_model(csv_path=args.csv, samples_per_class=args.samples)
