<div align="center">

# 🗳️ Election Analysis Dashboard

**An interactive Streamlit dashboard exploring Indian election voter turnout and political ad spend, with built-in statistical tests and ML models.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)

</div>

---

## 📖 Overview

This project analyses the relationship between political advertisement spending and voter turnout in Indian (Lok Sabha) elections. It combines constituency-level election results (electors, votes polled, turnout %, election phase) with party-level digital ad-spend data and state-level spend totals, merging them into a single dataset. Everything is served through a multi-tab Streamlit dashboard that mixes exploratory charts with statistical tests and machine-learning models — and every tab can export its findings as a downloadable PDF report.

## ✨ Features

- Sidebar filters: pick a political party (or "All") and search by state, with all tabs updating live
- KPI overview: total ad spend (INR), number of advertising parties, and top-3 spenders
- **Ad Spend by State** — bar chart of total advertisement spend per state
- **Voter Turnout** — average polled-percentage by state
- **Party Ad Spend** — pie chart of the top 5 parties by ad spend
- **Phase-wise Analysis** — dual-axis chart comparing ad spend and turnout across election phases
- **K-Means clustering** segmenting states by ad spend vs. turnout
- **Regression suite** — Linear Regression, SVR (RBF kernel), and Random Forest models predicting turnout from ad spend, reporting R² and MAE
- **Chi-Square test** of independence between state and election phase, with a contingency-table heatmap
- One-click **PDF report download** on every tab (generated with FPDF)

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Web app | Streamlit |
| Data wrangling | pandas |
| Visualisation | Plotly (Express + Graph Objects) |
| Machine learning | scikit-learn (KMeans, LinearRegression, SVR, RandomForestRegressor) |
| Statistics | SciPy (chi-square test) |
| Reporting | FPDF (PDF generation) |

## 📂 Project Structure

```text
Election-analysis/
├── election_dashboard.py   # The entire Streamlit dashboard (9 tabs)
├── results.csv             # Constituency-level results: state, PC name, electors, votes, turnout %, phase
├── advertisers.csv         # Party/page-level ad spend from the ad library (page, disclaimer, INR spent, ad count)
└── locations.csv           # State-level total ad spend (INR)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/Ankita-Bha/Election-analysis.git
cd Election-analysis
pip install streamlit pandas plotly scikit-learn scipy fpdf
```

### Usage

```bash
streamlit run election_dashboard.py
```

The app loads the three bundled CSVs from the repo root, merges results with state-level spend, and opens the dashboard in your browser. Use the sidebar to filter by party or search for a state, then explore the nine analysis tabs and download PDF reports.

## 📊 Results

The dashboard computes all statistics live from the bundled data rather than storing fixed results:

- Model tabs (Linear Regression, SVM, Random Forest) display **R² score and Mean Absolute Error** for predicting voter turnout from ad spend on the current filter selection
- The Chi-Square tab reports the test statistic and p-value with a plain-language significance interpretation
- The clustering tab visualises 3 K-Means clusters of states by spend vs. turnout
- Each tab exports a PDF report of its tables and metrics

## 🔮 Future Improvements

- Add multivariate models (more features than ad spend alone) for turnout prediction
- Replace per-run PDF writes with in-memory generation to avoid file clutter
- Add a choropleth map of India for spend and turnout
- Split the single-file dashboard into modules (data loading, models, UI)
- Pin dependencies in a `requirements.txt` and add caching for model training

## 👤 Author

**Ankita Bhamidimarri** — [@Ankita-Bha](https://github.com/Ankita-Bha)

---

<div align="center">
<sub>⭐ If you found this project useful, consider giving it a star!</sub>
</div>
