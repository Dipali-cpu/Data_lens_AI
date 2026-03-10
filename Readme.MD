# 🔍 DataLens AI — Dataset Analyser + LinkedIn Article Generator

A powerful Streamlit app that gives you deep dataset analysis and auto-generates LinkedIn articles using **Groq's ultra-fast LLaMA 3** inference.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

### 3. Open in browser

Streamlit will open automatically at `http://localhost:8501`

---

## 🔑 Groq API Key

1. Visit [console.groq.com](https://console.groq.com/)
2. Sign up / log in → Create API Key
3. Paste it into the sidebar of the app

---

## 📁 Supported File Formats

| Format  | Extension          |
| ------- | ------------------ |
| CSV     | `.csv`           |
| Excel   | `.xlsx`,`.xls` |
| JSON    | `.json`          |
| Parquet | `.parquet`       |
| TSV     | `.tsv`           |

---

## ✨ Features

### 📋 Overview Tab

* Live data preview with adjustable row count
* Column type inspector (numeric, categorical, datetime)
* Descriptive statistics table
* Missing value bar chart
* Categorical value frequency charts

### 📈 Distribution Tab

* Interactive histograms for all numeric columns
* Box plots for outlier detection
* Violin plots showing distribution shape
* Scatter plot explorer with OLS trendline
* Color encoding by categorical dimensions

### 🔗 Correlation Tab

* Full correlation heatmap (color-coded)
* Top N most correlated pairs ranked
* Interactive scatter matrix (pair plot)

### 🤖 AI Insights Tab (Groq)

* 6 preset analysis modes:
  * Executive Summary
  * Data Quality Report
  * Key Trends & Patterns
  * Anomalies & Outliers
  * Business Recommendations
  * Statistical Deep Dive
* Custom question interface
* Multi-turn chat with your dataset (memory-aware)

### 💼 LinkedIn Article Tab (Groq)

* Fully customisable article generation
* Tone: Professional / Conversational / Data-driven / Storytelling
* Target audience selection
* Optional emojis, CTA, hashtags
* Download as `.txt` or `.md`
* Shareable chart auto-generated for the post
* Viral LinkedIn tips included

---

## 🛠 Tech Stack

* **Frontend** : Streamlit
* **AI** : Groq API (LLaMA 3.3 70B)
* **Charts** : Plotly (interactive), Matplotlib, Seaborn
* **Data** : Pandas, NumPy

---

## 📸 Aesthetic

Dark editorial design with:

* Violet/Purple/Pink gradient accents
* Custom `Syne` + `Inter` typography
* Smooth hover animations
* Glassmorphism-style cards
