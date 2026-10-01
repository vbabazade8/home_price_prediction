# Home Price Prediction

Predicts apartment sale prices in Baku using a machine learning model
trained on real listings scraped from bina.az.

## Live demo

https://home-price-prediction-ae4m.onrender.com/static/index.html

Hosted on Render's free tier: the service sleeps after 15 minutes
without traffic, so the first request after that can take about a
minute while it wakes up.

## Project stages

1. **Scraping** (`scripts/scraper.py`) — collects the full sale catalog
   from bina.az's GraphQL API (`SearchItems` operation, ~58.7k listings).
   Pages of 25 listings (the server's maximum), cursor pagination,
   retries on network errors. Saves raw data to `data/items_all.csv`.

2. **Cleaning** (`scripts/clean_data.py`) — drops listings without rooms
   or district, unrealistic prices (< 5,000 AZN or < 300 AZN/m²), repeated
   ids, and the same property re-posted under different ids. Prints the
   row count after every step. Saves `data/item_clean_all.csv`
   (44,382 rows).

3. **Model development** (notebooks, in order):
   - `scripts/01_train_filtered_data.ipynb` — first model on ~950 VIP listings
   - `scripts/02_explore_train_full_data.ipynb` — 10x more data, Random Forest
   - `scripts/03_eda.ipynb` — exploratory data analysis
   - `scripts/04_model_selection.ipynb` — duplicates, full catalog,
     LazyPredict, Optuna, final model

4. **API** (`api/main.py`) — a FastAPI backend that loads the trained
   model and exposes `/predict` and `/locations` endpoints.

5. **Frontend** (`static/index.html`, `static/style.css`) — a form with a
   district dropdown that calls the API and shows the predicted price.

6. **Containerization** (`api/Dockerfile`) — packages the API, model,
   and frontend into a Docker image so it runs the same way on any
   machine.

7. **Deployment** — the Docker image is deployed on Render (free
   tier), giving the app a public URL.

## Model architecture

- **Algorithm:** `HistGradientBoostingRegressor` (scikit-learn), tuned
  with Optuna: learning_rate 0.038, 1,100 iterations, 222 leaves per
  tree, min_samples_leaf 5, max_features 0.54. Model size: ~27 MB.
- **Features (input to the model):**
  - `rooms` — number of rooms
  - `area` — total area (m²)
  - `floor` — floor the apartment is on
  - `floors` — total floors in the building
  - `hasRepair` — whether the apartment has renovations (0/1)
  - `isVipped`, `isFeatured` — whether the original listing was
    promoted on bina.az (fixed to a constant value in the API, not
    user-editable — see "Why these two features are fixed" below)
  - `location_*` — one-hot encoded district (83 columns; districts
    with fewer than 5 listings are grouped into `location_Other`)
- **Target:** `price` (sale price in AZN)

## Why this model

**Metrics.** MAE (mean absolute error, in AZN) is the primary metric:
"the model is off by this many manats on average". RMSE (penalizes large
errors), R² (share of price variance explained) and MAPE (error as % of
the real price) are reported alongside.

**1. Many models at once.** LazyPredict trained ~40 regressors on a
random 10k-row sample. Tree ensembles and boosting led; linear models
plateaued at R² ≈ 0.73. The ranking depended on the metric: boosting led
on R²/RMSE, ExtraTrees and Random Forest on MAE.

**2. Shortlist on all data** (5-fold cross-validation, default settings):

| Model | MAE | RMSE | R² | MAPE | Size |
|---|---|---|---|---|---|
| ExtraTrees | 36,046 | 94,515 | 0.834 | 11.4% | 482 MB |
| Random Forest | 37,598 | 96,404 | 0.827 | 11.8% | 318 MB |
| HistGradientBoosting | 47,115 | 108,129 | 0.783 | 14.9% | 0.4 MB |

The forests were the most accurate but far too large to deploy (GitHub
file limit 100 MB, Render free tier 512 MB RAM). So the goal became:
the most accurate model that fits these limits.

**3. Final comparison** on a 20% hold-out not used for tuning: a smaller
ExtraTrees (50 trees, min_samples_leaf 3) vs HistGradientBoosting tuned
with Optuna (40 trials, 3-fold CV):

| Model | MAE | RMSE | R² | MAPE | Size |
|---|---|---|---|---|---|
| ExtraTrees (small) | 40,216 | 114,876 | 0.779 | 12.2% | 56 MB |
| **HistGradientBoosting (tuned)** | **37,456** | **100,104** | **0.832** | **11.5%** | **27 MB** |

The tuned boosting model wins on every metric at half the size. The final
model is retrained on all 44k rows.

## How the model was improved

| Step | Data | Model | MAE (AZN) |
|---|---|---|---|
| First version | ~950 VIP listings | Random Forest | ~88,600 |
| 10x more data | ~9,500 listings | Random Forest | ~42,000 (too optimistic, see below) |
| Duplicates removed | ~8,600 listings | Random Forest | ~44,300 |
| Full catalog | ~44,400 listings | Random Forest | ~38,200 |
| Model choice + tuning | ~44,400 listings | HistGradientBoosting | ~37,500 (hold-out) |

- **Duplicates.** EDA found that ~10% of listings were the same property
  re-posted under different ids. Copies in both train and test made the
  test score too optimistic (data leakage). Removing them gave an honest
  baseline.
- **More data.** A learning curve showed MAE still falling as training
  data grew, so the full catalog was scraped. With the same model, MAE
  dropped from ~44.3k to ~38.2k.
- **Model choice and tuning** — see "Why this model".

Numbers in different rows come from different test sets, so they show
the direction and size of each improvement, not exact differences.

## Why these two features are fixed (isVipped, isFeatured)

VIP-promoted listings in the training data have a higher average
price per square meter than non-VIP listings, even accounting for
area and room count. However, this is very likely correlation from
self-selection — owners of pricier apartments more often choose to
pay for VIP promotion — rather than VIP status itself increasing an
apartment's value. Exposing this as a user-editable field in the form
would let someone artificially inflate their predicted price by
toggling it, without changing anything about the actual apartment. To
avoid this, both fields are fixed to a constant value inside the API
and are not part of the form.

## Known limitations

- **Baku only.** 98.4% of listings in the full catalog are in Baku
  (Xırdalan 612, Sumqayıt 83, other cities single listings), so
  predictions outside Baku are unreliable.
- **Repair is mixed up with new buildings.** In the data, "no repair"
  mostly means "new building sold without finishing" (e.g. Ağ şəhər,
  Sea Breeze), and new buildings are more expensive. The model can
  therefore predict a *lower* price with repair for some inputs (e.g. a
  250 m² apartment in Xətai). A new-build/resale feature would fix this.
- **Duplicate detection is approximate.** Listings are treated as
  duplicates when price, rooms, area, floor, floors, district and repair
  all match; two genuinely different identical apartments in the same
  building may be removed as well.
- The district dropdown lists only districts with 5+ listings.

## Running locally

Each part of the project has its own dependencies. See the
`requirements-*.txt` files in `scripts/` and `api/`.

**Scraper and cleaning:**

```
python -m venv scripts/.venv-scraper
scripts\.venv-scraper\Scripts\activate
pip install -r scripts/requirements-scraper.txt
python scripts/scraper.py
python scripts/clean_data.py
```

**Model training (Jupyter notebooks):**

```
python -m venv scripts/.venv-train
scripts\.venv-train\Scripts\activate
pip install -r scripts/requirements-train.txt
jupyter notebook scripts/04_model_selection.ipynb
```

**API (without Docker):**

```
python -m venv api/.venv-api
api\.venv-api\Scripts\activate
pip install -r api/requirements-api.txt
uvicorn api.main:app --reload
```

**API (with Docker):**

```
docker build -f api/Dockerfile -t home-price-api .
docker run -p 8000:8000 home-price-api
```