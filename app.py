"""
app.py — Phishing URL Detector
Upload a CSV of URLs → get back a labeled CSV (Phishing / Legitimate)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from train import train_model, extract_features

st.set_page_config(page_title="🎣 Phishing URL Detector", page_icon="🎣", layout="centered")
pd.set_option("styler.render.max_elements", 2000000)
st.title("🎣 Phishing URL Detector")
st.caption("Upload a CSV of URLs — get back a labeled result: **Phishing** or **Legitimate**.")

MODEL_PATH = "phishing_model.pkl"

# ── Train on first run ──────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    with st.spinner("Training model for the first time... (~5 seconds)"):
        train_model(MODEL_PATH)
    st.success("Model ready!")

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

clf = model_data["model"]

# ── CSV Format Instructions ─────────────────────────────────────────────────
with st.expander("📋 What should my CSV look like?"):
    st.markdown("""
Your CSV just needs **one column** containing URLs. The column can be named anything.

**Example:**
```
url
https://www.google.com
http://paypal-verify-account.xyz/login
https://github.com/user/repo
http://bit.ly/suspicious
```
You can also paste URLs with no header — the app handles both.
    """)
    example = pd.DataFrame({"url": [
        "https://www.google.com",
        "http://paypal-verify-account.xyz/login",
        "https://github.com/user/repo",
        "http://bit.ly/suspicious123",
    ]})
    st.download_button(
        "⬇️ Download example CSV",
        example.to_csv(index=False),
        file_name="example_urls.csv",
        mime="text/csv",
    )

# ── File Upload ─────────────────────────────────────────────────────────────
st.divider()
uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # Auto-detect URL column
    url_col = None
    for col in df.columns:
        if df[col].astype(str).str.contains("http|www\.", regex=True, na=False).mean() > 0.3:
            url_col = col
            break
    if url_col is None:
        url_col = df.columns[0]  # fallback to first column

    urls = df[url_col].astype(str).str.strip()
    st.info(f"Found **{len(urls)} URLs** in column `{url_col}`")

    with st.spinner("Analyzing URLs..."):
        features = np.array([extract_features(u) for u in urls])
        predictions = clf.predict(features)
        labels = ["🚨 Phishing" if p == 1 else "✅ Legitimate" for p in predictions]

    # ── Results ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Results")

    result_df = pd.DataFrame({
        "URL": urls.values,
        "Result": labels,
    })

    phishing_count = sum(1 for l in labels if "Phishing" in l)
    legit_count = len(labels) - phishing_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Total URLs", len(labels))
    col2.metric("🚨 Phishing", phishing_count)
    col3.metric("✅ Legitimate", legit_count)

    st.divider()

    # Color-code the table
    def highlight_row(row):
        color = "#ffe5e5" if "Phishing" in row["Result"] else "#e5ffe5"
        return [f"background-color: {color}"] * len(row)
    styled_df = result_df.style.apply(highlight_row, axis=1)
    st.dataframe(styled_df, use_container_width=True)

    # ── Download Results ───────────────────────────────────────────────────
    st.divider()
    clean_result = pd.DataFrame({
        "URL": urls.values,
        "Label": ["Phishing" if p == 1 else "Legitimate" for p in predictions],
    })
    st.download_button(
        "⬇️ Download labeled CSV",
        clean_result.to_csv(index=False),
        file_name="phishing_results.csv",
        mime="text/csv",
    )

    if phishing_count > 0:
        st.warning(f"⚠️ {phishing_count} suspicious URL(s) detected. Do not visit these links.")

else:
    st.info("👆 Upload a CSV file above to get started.")
