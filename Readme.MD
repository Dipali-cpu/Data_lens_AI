# 🔍 DataLens AI

> Upload any dataset. Get deep analysis + AI insights + LinkedIn articles — in seconds.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Click_Here-238636?style=for-the-badge)](https://datalensai-gvkkrxx8jkdlsxuph8kczh.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-e85c0d?style=for-the-badge)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-2ea043?style=for-the-badge)](LICENSE)

---

## 🌐 Live Demo

**[👉 Try DataLens AI now →](https://datalensai-gvkkrxx8jkdlsxuph8kczh.streamlit.app/)**

No installation needed. Upload any CSV, Excel, JSON, Parquet, or TSV file and explore instantly.

---

## 💡 What is DataLens AI?

DataLens AI is a **full-stack data analytics web app** that turns any raw dataset into actionable insights within seconds. Built for analysts, data scientists, and business teams who want to skip the setup and go straight to understanding their data.

> No code needed. Upload → explore → get AI-powered insights → generate a LinkedIn article ready to post.

**Why I built this:** I wanted a single tool that combines EDA, AI analysis, and content generation — saving hours of manual work for every new dataset.

---

## ✨ Features

| Tab | What it does |
|-----|-------------|
| 📋 **Overview** | Live data preview, column inspector, missing value chart, categorical breakdown |
| 📈 **Distributions** | Histograms, box plots, violin plots, scatter explorer with OLS trendline |
| 🔗 **Correlations** | Full heatmap, top N correlated pairs ranked, interactive pair plot |
| 🤖 **AI Insights** | 6 analysis modes + multi-turn chat with your data (Groq LLaMA 3.3 70B) |
| 💼 **LinkedIn Article** | Auto-generate viral-ready posts with tone, audience & hashtag controls |

---

## 🤖 AI Analysis Modes

- 📊 Executive Summary
- 🔍 Data Quality Report
- 📈 Key Trends & Patterns
- ⚠️ Anomalies & Outliers
- 💡 Business Recommendations
- 🧠 Statistical Deep Dive
- 💬 Custom question — ask anything about your data

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit (custom dark-mode CSS) |
| AI Engine | Groq API → LLaMA 3.3 70B Versatile |
| Charts | Plotly (interactive), Matplotlib, Seaborn |
| Data | Pandas, NumPy |
| Deployment | Streamlit Community Cloud |

---

## 📁 Supported File Formats

`CSV` · `Excel (.xlsx / .xls)` · `JSON` · `Parquet` · `TSV`

---

## 🚀 Run Locally

```bash
git clone https://github.com/Dipali-cpu/Data_lens_AI.git
cd Data_lens_AI
pip install -r requirements.txt
```

Add your Groq API key (free at [console.groq.com](https://console.groq.com)):
```bash
# Create a .env file
echo "GROQ_API_KEY=your_key_here" > .env
```

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
Data_lens_AI/
├── app.py              # Main Streamlit application (1,358 lines)
├── requirements.txt    # Python dependencies
├── .gitignore
└── Readme.MD
```

---

## 🎯 Key Technical Decisions

- **Groq over OpenAI** — Groq's inference is 10–20x faster, making the chat feel instant
- **Session-based chat memory** — `st.session_state` stores conversation history for multi-turn Q&A
- **Secrets management** — reads from `st.secrets` (cloud) with `.env` fallback (local), so the API key is never exposed
- **5 file formats** — handled with a single `load_data()` router function

---

## 👩‍💻 Built by

**Dipali** — Data Science & Analytics professional  
Self-taught in Python, Machine Learning, SQL, and AI-integrated applications.  
Completed Data Science internship · April 2026

[![LinkedIn](https://img.shields.io/badge/Connect_on-LinkedIn-0077b5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/dipali-chothmal-5731b032b/)
[![GitHub](https://img.shields.io/badge/More_Projects-GitHub-333?style=flat-square&logo=github)](https://github.com/Dipali-cpu)

---

## 📄 License

MIT License — free to use, modify, and distribute.
