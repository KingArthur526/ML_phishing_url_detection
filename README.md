# 🎣 Phishing URL Detector

Upload a CSV of URLs → get back a labeled result: **Phishing** or **Legitimate**.

## Setup & Run

```bash
pip install -r requirements.txt

# Option A: train on the real dataset (recommended, best accuracy)
python3 train.py --csv malicious_phish.csv

# Option B: run without a dataset (trains on built-in synthetic data, ~5 sec)
streamlit run app.py
```

## Accuracy (trained on real data)

| Class | Accuracy |
|---|---|
| Phishing | 89.0% |
| Benign | 91.5% |
| Defacement | 98.5% |
| Malware | 95.5% |
| **Overall** | **93.52%** |

Trained on 60,000 real URLs from the malicious_phish.csv dataset (651k total).

## Central Algorithm

**URL Feature Engineering + Random Forest**

20 features extracted from the raw URL string — no network calls:

| Feature Group | Examples |
|---|---|
| Length-based | URL length, hostname length, path length |
| Character counts | Dots, hyphens, `@` symbols, `%` encoding |
| Structural signals | HTTPS, subdomain depth, IP address as host |
| Keyword signals | Phishing keywords, URL shortener, short domain |

→ 300-tree Random Forest classifies the feature vector as Phishing or Not Phishing.

## Project Structure

```
phishing_detector/
├── app.py              # Streamlit UI
├── train.py            # Feature engineering + Random Forest
├── requirements.txt
└── README.md
```
