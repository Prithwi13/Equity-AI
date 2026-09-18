<div align="center">

<br />

<h1>📈 EquityAI</h1>
<h3>Agentic AI-Powered Equity Research Analyst</h3>

<br />

<p>
  <a href="https://colab.research.google.com/github/Prithwi13/Equity-AI/blob/main/6302_final_project.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FinBERT-ProsusAI-FF6F00?style=flat-square&logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-Champion_Model-EC6C00?style=flat-square" />
  <img src="https://img.shields.io/badge/scikit--learn-TimeSeries_CV-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Alpha_Vantage-OHLCV-3DDC84?style=flat-square" />
  <img src="https://img.shields.io/badge/MarketAux-News_Feed-1A73E8?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=flat-square" />
</p>

<br />

> **EquityAI** is a notebook-based equity research pipeline. It ingests 5-minute OHLCV price bars and financial news headlines, scores sentiment using **FinBERT** (a BERT model fine-tuned on financial text), engineers technical and sentiment-decay features, and compares models with time-series cross-validation before producing a structured research note.

<br />

```
 Raw News Headlines ──► FinBERT Scoring ──► News Decay / Forward-Fill ──┐
                                                                          ├──► Merged Feature Matrix ──► GridSearchCV ──► Champion Model ──► Report
 5-min OHLCV Bars ───► Incremental Store ──► RSI · MACD · Bollinger ────┘
```

</div>

---

## 📋 Table of Contents

1. [What Is EquityAI?](#-what-is-equityai)
2. [System Architecture](#-system-architecture)
3. [Pipeline — Stage by Stage](#-pipeline--stage-by-stage)
   - [Stage 1 · Incremental Data Ingestion](#stage-1--incremental-data-ingestion)
   - [Stage 2 · FinBERT Sentiment Scoring](#stage-2--finbert-sentiment-scoring)
   - [Stage 3 · Feature Engineering & News Decay](#stage-3--feature-engineering--news-decay)
   - [Stage 4 · Model Training & Champion Selection](#stage-4--model-training--champion-selection)
   - [Stage 5 · Agentic Report Generation](#stage-5--agentic-report-generation)
4. [Tech Stack](#-tech-stack)
5. [Data Schemas](#-data-schemas)
6. [Model Design Decisions](#-model-design-decisions)
7. [Project Structure](#-project-structure)
8. [Quickstart](#-quickstart)
9. [Environment & API Keys](#-environment--api-keys)
10. [Configuration Reference](#-configuration-reference)
11. [Roadmap](#-roadmap)
12. [Contributing](#-contributing)
13. [License](#-license)

---

## 🔭 What Is EquityAI?

EquityAI connects five parts of a short-horizon research workflow, from raw data collection to a generated research note:

```
Data Ingestion  →  Sentiment Enrichment  →  Feature Engineering  →  Model Selection  →  Agentic Report
```

**Covered tickers:** `AAPL` · `MSFT` · `GOOGL` · `AMZN` · `TSLA`

**Prediction target:** Binary direction — will the stock price be **higher** in the next 5 minutes? (`1 = UP`, `0 = NOT UP`)

### What makes it different

| Property | Most Sentiment Tools | EquityAI |
|---|---|---|
| NLP model | Generic (VADER, TextBlob) | **FinBERT** — finance-domain fine-tuned BERT |
| Data storage | Re-fetch every run | **Incremental** — only appends new records |
| Time handling | Naive train/test split | **TimeSeriesSplit** — zero look-ahead leakage |
| Sentiment gaps | Fill with 0 (neutral) | **Forward-fill decay** — news persists until overwritten |
| Output | Raw predictions | **Agentic narrative report** from live web data |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              EquityAI  —  Full System                           │
├─────────────────┬───────────────────┬──────────────────┬────────────────────────┤
│  STAGE 1        │  STAGE 2          │  STAGE 3 & 4     │  STAGE 5               │
│  DATA INGEST    │  SENTIMENT NLP    │  ML PIPELINE     │  AGENTIC REPORT        │
│                 │                   │                  │                        │
│ ┌─────────────┐ │ ┌───────────────┐ │ ┌──────────────┐ │ ┌────────────────────┐ │
│ │StockPriceAPI│ │ │FinBERT        │ │ │Merge + Resamp│ │ │  Live Web Scrape   │ │
│ │Alpha Vantage│ │ │ProsusAI/      │ │ │5-min bins    │ │ │  Sentiment Summary │ │
│ │5-min OHLCV  │ │ │finbert        │ │ │Forward-fill  │ │ │  Prediction Output │ │
│ └──────┬──────┘ │ │batch_size=32  │ │ │Decay Model   │ │ │  Feature Insights  │ │
│        │        │ └───────┬───────┘ │ └──────┬───────┘ │ └────────────────────┘ │
│ ┌─────────────┐ │         │         │        │          │                        │
│ │MarketAuxAPI │ │ compound│score    │ ┌──────▼───────┐ │                        │
│ │News Headlines│ │ [-1, +1]│        │ │Feature Eng.  │ │                        │
│ │per ticker   │ │         │         │ │RSI·MACD·BB   │ │                        │
│ └──────┬──────┘ │         │         │ │Lags·Rolling  │ │                        │
│        │        │         │         │ │Time features │ │                        │
│ ┌─────────────┐ │         │         │ └──────┬───────┘ │                        │
│ │Incremental  │ │         │         │        │          │                        │
│ │Storage      │ │         │         │ ┌──────▼───────┐ │                        │
│ │Dedup+Append │ │         │         │ │GridSearchCV  │ │                        │
│ │prices_      │ │         │         │ │TimeSeriesSpl.│ │                        │
│ │master.csv   │ │         │         │ │RF vs XGBoost │ │                        │
│ │news_        │ │         │         │ └──────┬───────┘ │                        │
│ │master.csv   │ │         │         │        │          │                        │
│ └─────────────┘ │         │         │ Champion Model   │                        │
│                 │         │         │  final_model     │                        │
└─────────────────┴─────────┴─────────┴──────────────────┴────────────────────────┘
```

---

## 🔄 Pipeline — Stage by Stage

### Stage 1 · Incremental Data Ingestion

**Key classes:** `StockPriceAPI`, `MarketAuxAPI`, `IncrementalDataStorage`, `DataCollector`

The ingestion layer pulls two independent data streams per ticker and stores them in a self-healing, deduplication-safe CSV store.

#### Price Data — Alpha Vantage

- **Endpoint:** `TIME_SERIES_INTRADAY`
- **Resolution:** 5-minute OHLCV bars
- **Timezone:** All timestamps are normalized to **UTC** (localized from `America/New_York`)
- **Output size:** `compact` for same-day, `full` for multi-day / backfill runs
- **Rate limiting:** Built-in `time.sleep(13)` between ticker calls — stays within the free tier's 5 calls/minute ceiling

#### News Data — MarketAux

- Financial headlines with `publishedAt`, source, description, and URL per ticker
- Company-name query mapping (`AAPL → "Apple"`, `GOOGL → "Google"`, etc.) for broader article coverage
- Configurable lookback window via `news_days_back`

#### `IncrementalDataStorage` — The Core Storage Engine

This class is what makes the pipeline reliable and re-runnable without wasting API quota:

```
First run  →  Create prices_master.csv + news_master.csv from scratch
Daily run  →  Load existing files → Append new records only → Deduplicate → Re-sort → Save
Backfill   →  Overwrite mode — replaces existing data with historical backfill
```

**Deduplication keys:**
- Prices: `(timestamp, ticker)` — prevents duplicate 5-min bars
- News: `(headline, ticker)` — prevents the same article appearing twice

```python
# One-time historical backfill (5 years)
main_data_collection('backfill')

# Daily incremental delta update
main_data_collection('incremental')
```

**Storage output:**

| File | Contents |
|---|---|
| `data/raw/prices_master.csv` | 21,120 rows · 5 tickers · 5-min OHLCV bars |
| `data/raw/news_master.csv` | 458 articles · 5 tickers · headlines + metadata |
| `data/raw/daily_backups/` | Auto-timestamped backup copies |

---

### Stage 2 · FinBERT Sentiment Scoring

**Model:** [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert) — a BERT-base model fine-tuned on financial phrases, earnings call transcripts, and financial news corpora.

#### Why FinBERT over VADER or TextBlob?

Generic sentiment models fail on financial language. The word "beats" is positive in finance but ambiguous in general text. "Headwinds", "dilution", "covenant breach", "guidance raised" — these carry strong directional signals that only a finance-tuned model understands correctly.

FinBERT outputs three class probabilities: **positive**, **negative**, **neutral**. The pipeline collapses these into a single compound score:

```
compound_score  =  P(positive) − P(negative)  ∈  [−1.0, +1.0]
```

This produces a continuous, signed sentiment signal ready for direct use as an ML feature.

#### Efficiency Design

Scoring hundreds of headlines naively would be slow. The pipeline uses three optimisations:

1. **Deduplicate first** — only unique headlines are scored; results are then joined back via `pd.merge` on the full dataset
2. **Batched inference** — processes 32 headlines per forward pass
3. **`@torch.no_grad()`** — disables gradient tracking, cutting memory and compute significantly

```python
# Batch inference loop
for i in range(0, len(unique_headlines), batch_size=32):
    batch = unique_headlines[i : i + 32]
    scores = get_finbert_sentiment(batch, tokenizer, model)
```

**Output:** `news_with_finbert_sentiment.csv` — identical to `news_master.csv` with one additional column: `sentiment` (float, compound score per headline).

**Observed sentiment by ticker (actual data):**

| Ticker | Mean Sentiment | Signal |
|---|---|---|
| `MSFT` | +0.040 | Mildly positive |
| `GOOGL` | +0.031 | Mildly positive |
| `AAPL` | +0.030 | Mildly positive |
| `AMZN` | −0.023 | Slight negative bias |
| `TSLA` | **−0.194** | Strongly negative coverage |

---

### Stage 3 · Feature Engineering & News Decay

This stage is where the two data streams are merged and transformed into the feature matrix fed to the ML models.

#### Step 1 · Resample News into 5-Minute Bins

News arrives at irregular timestamps (e.g., `09:31:47`, `09:43:02`). Price data sits on a clean 5-minute grid (`09:30`, `09:35`, `09:40`…). The pipeline bins all news into matching 5-minute intervals using `pandas.resample('5T')`:

```python
ticker_agg = group.resample('5T').agg(
    sentiment_mean=('sentiment', 'mean'),   # average FinBERT score in that window
    news_count=('headline', 'count')        # number of articles (signal strength)
)
```

#### Step 2 · Left-Merge on Price Grid

A `left` merge preserves every price bar, producing `NaN` for any 5-minute window with no news. This is intentional — the NaNs are handled semantically in the next step.

#### Step 3 · News Decay via Forward-Fill

This is the most important preprocessing decision in the pipeline. Most 5-minute windows contain no news, but that doesn't mean the market has "forgotten" the last headline it saw. **Sentiment persists until new information arrives.** This is modelled with `ffill()` (forward-fill), applied within each ticker group:

```python
# CRITICAL: sort before filling — ensures signal propagates forward in time
merged_df = merged_df.sort_values(by=['ticker', 'timestamp'])

# Carry the last known sentiment score forward
merged_df['sentiment_mean'] = merged_df.groupby('ticker')['sentiment_mean'].ffill()

# Bars before the first ever news article get score = 0 (neutral)
merged_df['sentiment_mean'] = merged_df['sentiment_mean'].fillna(0)
```

Meanwhile, `news_count` uses `fillna(0)` — a bar with no articles literally had zero coverage.

#### Step 4 · Full Feature Engineering

| Category | Features | Description |
|---|---|---|
| **Raw Price** | `open`, `high`, `low`, `close`, `volume` | Direct OHLCV inputs |
| **Sentiment** | `sentiment_mean`, `news_count` | FinBERT compound + article count (per 5-min bin) |
| **Price Lag** | `close_lag_1` | Price from the immediately prior bar |
| **Sentiment Lag** | `sentiment_lag_1` | Sentiment from the immediately prior bar |
| **Rolling Mean (1hr)** | `close_rolling_12`, `sentiment_rolling_12` | 12-bar (60-min) rolling average of price and sentiment |
| **Time** | `hour`, `minute` | Intraday session effects (open / close / lunch) |
| **RSI (14)** | `rsi` | Relative Strength Index — momentum overbought/oversold signal |
| **MACD (12/26/9)** | `macd`, `macd_signal` | Moving Average Convergence/Divergence crossover signal |
| **Bollinger Bands (20, 2σ)** | `bollinger_upper`, `bollinger_lower` | Volatility envelope |
| **Ticker** | `ticker_MSFT`, `ticker_TSLA`, … | One-hot encoded ticker identity |

All technical indicators are computed **per-ticker** inside a `groupby` loop to prevent cross-contamination between tickers' MACD/RSI histories.

#### Target Variable

```python
# "Will the price be higher in exactly 5 minutes?"
df_model['future_close'] = df_model.groupby('ticker')['close'].shift(-1)
df_model['target_direction'] = (df_model['future_close'] > df_model['close']).astype(int)
# 1 = UP, 0 = FLAT or DOWN
```

---

### Stage 4 · Model Training & Champion Selection

Three models compete, evaluated under conditions that mirror real-world deployment.

#### Model 1 — Logistic Regression (Balanced Baseline)

The baseline establishes a floor. It uses `class_weight='balanced'` — a critical fix to prevent the model from exploiting any slight class imbalance by always predicting the majority class.

```python
LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
```

Data is scaled with `StandardScaler` before fitting. The train/test split uses `shuffle=False` to respect temporal ordering.

#### Model 2 — GridSearchCV + TimeSeriesSplit (Champion Selection)

This is the core of the methodology. A scikit-learn `Pipeline` wraps `StandardScaler` + model, and `GridSearchCV` exhaustively searches over both model families simultaneously using `TimeSeriesSplit` with **n_splits=5** — guaranteeing that every validation fold is always in the future relative to its training fold:

```
Fold 1:  ██████░░░░░░░░░░░░░░   Train Jan–Feb  │ Validate Mar
Fold 2:  ████████░░░░░░░░░░░░   Train Jan–Mar  │ Validate Apr
Fold 3:  ██████████░░░░░░░░░░   Train Jan–Apr  │ Validate May
Fold 4:  ████████████░░░░░░░░   Train Jan–May  │ Validate Jun
Fold 5:  ██████████████░░░░░░   Train Jan–Jun  │ Validate Jul
```

**Search grid:**

```python
param_grid = [
    {   # Random Forest
        'model': [RandomForestClassifier(class_weight='balanced')],
        'model__n_estimators': [100, 200],
        'model__max_depth': [5, 10]
    },
    {   # XGBoost
        'model': [XGBClassifier(eval_metric='logloss')],
        'model__n_estimators': [100, 200],
        'model__max_depth': [3, 5],
        'model__learning_rate': [0.01, 0.1]
    }
]
```

The winning combination across all 5 folds is saved as `final_model` and evaluated on a held-out test set with a full `classification_report` (precision, recall, F1).

#### Feature Importance Analysis

The champion model's `feature_importances_` (Gini importance for RF; gain for XGBoost) are plotted as a ranked bar chart. This answers the key research question: **did FinBERT sentiment provide a genuine predictive signal above and beyond price technicals?** Look for `sentiment_mean`, `sentiment_lag_1`, or `sentiment_rolling_12` in the top features.

> A cross-validated accuracy above **53–55%** on 5-minute intraday data represents a meaningful, tradeable edge — short-horizon price movement is near-random, and even small consistent edges compound.

---

### Stage 5 · Agentic Report Generation

The final stage is an agentic system that autonomously:

1. **Fetches live data** from financial information websites
2. **Pulls the champion model's predictions** per ticker
3. **Reads the FinBERT sentiment narrative** — which headlines drove the score, and in which direction
4. **Surfaces the top feature importances** — what signal was most predictive this session
5. **Generates a structured research report** with a natural-language summary paragraph per ticker

The result is a self-contained equity research document — directional predictions, sentiment context, and model-driven insights — produced entirely without human authorship.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Price Data** | [Alpha Vantage API](https://www.alphavantage.co/) | 5-min intraday OHLCV bars |
| **News Data** | [MarketAux API](https://www.marketaux.com/) | Financial headlines per ticker |
| **Storage** | CSV (incremental append) | Dedup-safe, stateful data store |
| **NLP Inference** | [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) | Finance-domain sentiment scoring |
| **DL Framework** | PyTorch + HuggingFace Transformers | FinBERT model loading and batch inference |
| **Baseline Model** | scikit-learn `LogisticRegression` | Balanced-class linear baseline |
| **Champion Models** | `XGBClassifier`, `RandomForestClassifier` | Gradient-boosted tree ensembles |
| **Validation** | `TimeSeriesSplit` + `GridSearchCV` | Leakage-free chronological CV |
| **Pipeline** | `sklearn.pipeline.Pipeline` | Scaler + model in a single estimator |
| **Feature Eng.** | pandas, NumPy | Lags, rolling windows, RSI, MACD, Bollinger |
| **Visualisation** | Matplotlib, Seaborn | Correlation heatmap, feature importance plots |
| **Runtime** | Google Colab (GPU) | GPU-accelerated FinBERT inference |
| **Language** | Python 3.10+ | Full pipeline |

---

## 📊 Data Schemas

### `prices_master.csv`

| Column | Type | Description |
|---|---|---|
| `timestamp` | `datetime64[UTC]` | 5-minute bar open timestamp (UTC) |
| `ticker` | `str` | Equity symbol — AAPL, MSFT, GOOGL, AMZN, TSLA |
| `open` | `float64` | Bar open price |
| `high` | `float64` | Bar high price |
| `low` | `float64` | Bar low price |
| `close` | `float64` | Bar close price |
| `volume` | `int64` | Shares traded in the bar |

> **21,120 rows** · 5 tickers · Oct 20 – Nov 19, 2025 · 5-minute resolution

---

### `news_master.csv`

| Column | Type | Description |
|---|---|---|
| `timestamp` | `datetime64[UTC]` | Article publish time (UTC) |
| `headline` | `str` | Article title |
| `description` | `str` | Article summary/lede |
| `source` | `str` | Publisher name (e.g., Cult of Mac, 9to5Mac) |
| `url` | `str` | Canonical article URL |
| `ticker` | `str` | Associated equity symbol |

> **458 rows** · 5 tickers · Mixed sources including Nature.com, Biztoc, Yahoo Entertainment

---

### `news_with_finbert_sentiment.csv`

All columns from `news_master.csv`, plus:

| Column | Type | Description |
|---|---|---|
| `sentiment` | `float64` | FinBERT compound score: `P(pos) − P(neg)` ∈ `[−1, +1]` |

> Score range in dataset: `−0.962` to `+0.922` · Mean: `−0.022` (slight overall negativity across all tickers)

---

## 🧠 Model Design Decisions

**Why not shuffle the train/test split?**
Stock prices are a time series. Shuffling would allow the model to train on data from the future and test on the past — a classic form of look-ahead bias that produces falsely inflated accuracy metrics. All splits use `shuffle=False`.

**Why `class_weight='balanced'` on the baseline?**
Without it, a logistic regression on a slightly imbalanced target will learn to simply predict the majority class 100% of the time, achieving a misleadingly decent accuracy while being completely useless. The balanced weight forces equal attention to both directional classes.

**Why a `Pipeline` for GridSearchCV?**
Fitting `StandardScaler` on the full dataset before cross-validation would leak statistics (mean, std) from the validation folds into the scaler. Wrapping scaler + model in a `Pipeline` ensures the scaler is re-fitted on training data only at each CV fold.

**Why FinBERT over a rule-based system?**
Financial language is highly domain-specific and context-dependent. Rule-based systems assign fixed sentiment to words — they cannot understand that "Apple beats estimates" is very different from "Apple faces headwinds." FinBERT was trained to understand this context.

**Why forward-fill and not zero-fill for missing sentiment?**
Zero-fill assumes markets immediately forget news and return to neutral sentiment the moment a headline is published. That is empirically false. News sentiment decays gradually. Forward-filling is a simple but honest model of information persistence: the last known sentiment remains active until replaced by fresher news.

---

## 📁 Project Structure

```
EquityAI/
│
├── 6302_final_project.ipynb          # Master notebook — full pipeline, Colab-ready
│
├── data/
│   └── raw/
│       ├── prices_master.csv         # Incrementally-built OHLCV store
│       ├── news_master.csv           # Incrementally-built news store
│       └── daily_backups/            # Auto-created timestamped backups
│
├── news_with_finbert_sentiment.csv   # FinBERT-scored news output
│
└── README.md
```

**Notebook cell map:**

| Cells | Stage | What happens |
|---|---|---|
| 1–3 | Setup | Imports, Colab Secrets loading, error handling |
| 4–6 | API Classes | `StockPriceAPI` + `MarketAuxAPI` definitions |
| 7–9 | Storage Classes | `IncrementalDataStorage` + `DataCollector` definitions |
| 10–12 | Init | Instantiate all components with live API keys |
| 13–15 | Pre-check | `get_statistics()` — status before collection |
| 16–18 | Collection | `main_data_collection('backfill' | 'incremental')` |
| 19–21 | Post-check | Statistics verification after collection |
| 22–24 | Viewer | Sample data inspection helper |
| 25–26 | Standalone | Full self-contained script (env var compatible, no Colab dependency) |
| 27 | FinBERT | Sentiment scoring pipeline — batch inference, dedup, merge, save |
| 28 | ML Pipeline | Full modeling: merge → resample → forward-fill → features → RSI/MACD/BB → GridSearchCV → evaluation → feature importance |

---

## ⚡ Quickstart

### Option A — Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Prithwi13/6302_stock/blob/main/6302_final_project.ipynb)

1. Click the badge to open in Colab
2. In the left sidebar, click the **🔑 Secrets** tab
3. Add `ALPHAVANTAGE_KEY` and `MARKETAUX_KEY` with **Notebook access** toggled **ON**
4. Run all cells top to bottom — the notebook is self-explanatory at each step

---

### Option B — Local / Jupyter

```bash
# 1. Clone the repository
git clone https://github.com/Prithwi13/6302_stock.git
cd 6302_stock

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Export API keys as environment variables
export ALPHAVANTAGE_KEY=your_key_here
export MARKETAUX_KEY=your_key_here

# 5. Launch the notebook
jupyter notebook 6302_final_project.ipynb
```

---

### First Run: Seed the Data Store

On the very first run, execute the backfill to build a historical data foundation. This seeds `prices_master.csv` and `news_master.csv` with up to 5 years of price history and several weeks of headlines.

```python
# In the notebook — run once
main_data_collection('backfill')
```

> ⏱️ **Expected runtime:** ~65 seconds minimum due to the 13-second rate-limit pause per ticker (5 tickers × 13s). Historical backfills may take several minutes.

---

### Daily Updates

After the initial backfill, run incremental updates each trading day. Only new records are fetched and appended:

```python
main_data_collection('incremental')
```

---

### Run the Full ML Pipeline

After data is collected and sentiment is scored, execute **Cell 28** to run the complete modeling pipeline:

```
Merge → Resample → Forward-Fill → Feature Engineering → GridSearchCV → Report
```

---

## 🔑 Environment & API Keys

| Secret | Env Variable | Free Tier | Registration |
|---|---|---|---|
| Alpha Vantage | `ALPHAVANTAGE_KEY` | 25 req/day · 5 req/min | [alphavantage.co](https://www.alphavantage.co/support/#api-key) |
| MarketAux | `MARKETAUX_KEY` | 100 req/month | [marketaux.com](https://www.marketaux.com/) |

**In Google Colab:** Use the **🔑 Secrets** sidebar tab — keys are injected via `google.colab.userdata.get()`.

**Locally:** Export as shell environment variables or add to a `.env` file (which is excluded by `.gitignore`).

> ⚠️ **Never hardcode API keys.** The pipeline loads them at runtime only. No key is ever written to disk or committed to version control.

---

## ⚙️ Configuration Reference

All key parameters are defined at the top of their respective pipeline stage. Change these to adapt the system to your use case:

| Parameter | Default | Stage | Description |
|---|---|---|---|
| `TICKERS` | `['AAPL','MSFT','GOOGL','TSLA','AMZN']` | Ingest | Equities to track |
| `interval` | `'5m'` | Ingest | Price bar resolution |
| `backfill_start_date` | 5 years ago from today | Ingest | Historical start date for backfill |
| `news_days_back` | `30` | Ingest | News lookback window (days) |
| `MODEL_NAME` | `ProsusAI/finbert` | Sentiment | HuggingFace model ID |
| `batch_size` | `32` | Sentiment | FinBERT inference batch size |
| `resample_freq` | `'5T'` | Features | News binning frequency |
| `rolling_window` | `12` | Features | Rolling mean window (12 × 5min = 1hr) |
| `rsi_window` | `14` | Features | RSI period |
| `macd_short/long/signal` | `12 / 26 / 9` | Features | Standard MACD parameters |
| `bollinger_window` | `20` | Features | Bollinger Band period |
| `test_size` | `0.2` | Modeling | Held-out test set proportion |
| `tscv n_splits` | `5` | Modeling | TimeSeriesSplit folds |

---

## 🛣️ Roadmap

- [ ] **Streamlit dashboard** — live per-ticker sentiment, prediction confidence, and feature importance UI
- [ ] **LSTM / Transformer model** — replace tree ensembles with a sequence-aware architecture that exploits temporal structure natively
- [ ] **Options flow integration** — add implied volatility (IV rank) and put/call ratio as additional features
- [ ] **Expanded universe** — S&P 500 coverage via bulk API and parallel ticker collection
- [ ] **Automated daily scheduler** — GitHub Actions or Cloud Scheduler workflow to trigger incremental runs at market open
- [ ] **Backtesting module** — simulate portfolio P&L by translating directional predictions into long/short positions
- [ ] **FinBERT fine-tuning** — domain-adapt on earnings call transcripts and SEC 8-K filings for higher sentiment precision
- [ ] **PDF report export** — render the agentic analyst report as a downloadable, professionally formatted PDF

---

## 🤝 Contributing

Contributions are welcome. Please open an issue first to discuss major changes before submitting a pull request.

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "feat: describe your change clearly"
git push origin feature/your-feature-name
# Open a Pull Request against main
```

Please include a working notebook demo or unit test for any new pipeline stage.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

<br />

Built for **COMP 6302** &nbsp;·&nbsp; Powered by FinBERT, XGBoost & Alpha Vantage

*"Finding signal in the noise of financial markets."*

<br />

</div>
