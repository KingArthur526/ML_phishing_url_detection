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
    st.subheader("📊 Analysis Summary")

    phishing_count = sum(1 for l in labels if "Phishing" in l)
    legit_count = len(labels) - phishing_count
    total = len(labels)

    phishing_percent = (phishing_count / total) * 100
    legit_percent = (legit_count / total) * 100

    # ── Metrics ────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    col1.metric("Total URLs", total)
    col2.metric("🚨 Phishing", f"{phishing_percent:.1f}%")
    col3.metric("✅ Legitimate", f"{legit_percent:.1f}%")

    st.divider()

    # ── Percentage Visualization ──────────────────────────────────────────
    st.markdown("### 📈 URL Safety Distribution")

    st.write(f"🚨 Phishing URLs: {phishing_count} / {total}")
    st.progress(phishing_percent / 100)

    st.write(f"✅ Legitimate URLs: {legit_count} / {total}")
    st.progress(legit_percent / 100)

    # ── Distribution Chart ────────────────────────────────────────────────
    st.markdown("### 📊 Detection Overview")

    chart_data = pd.DataFrame({
        "Category": ["Phishing", "Legitimate"],
        "Count": [phishing_count, legit_count]
    })

    st.bar_chart(chart_data.set_index("Category"))

    # ── Example URLs ──────────────────────────────────────────────────────
    st.markdown("### 🔍 Example URLs")

    phishing_examples = [
        urls.values[i]
        for i, p in enumerate(predictions)
        if p == 1
    ]

    legit_examples = [
        urls.values[i]
        for i, p in enumerate(predictions)
        if p == 0
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.error("🚨 Example Phishing URL")

        if phishing_examples:
            st.code(phishing_examples[0])
        else:
            st.write("No phishing URLs detected.")

    with col2:
        st.success("✅ Example Legitimate URL")

        if legit_examples:
            st.code(legit_examples[0])
        else:
            st.write("No legitimate URLs found.")

    # ── Optional Preview Table ────────────────────────────────────────────
    with st.expander("📄 Preview first 10 analyzed URLs"):
        preview_df = pd.DataFrame({
            "URL": urls.values,
            "Result": labels,
        })

        st.dataframe(
            preview_df.head(10),
            use_container_width=True
        )

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

    # ── Final Warning ─────────────────────────────────────────────────────
    if phishing_count > 0:
        st.warning(
            f"⚠️ {phishing_count} suspicious URL(s) detected "
            f"({phishing_percent:.1f}% of uploaded URLs). "
            "Avoid visiting suspicious links."
        )
    else:
        st.success("🎉 No phishing URLs detected!")

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
