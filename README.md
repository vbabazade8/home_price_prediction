# Home Price Prediction

Predicts apartment sale prices in Baku using a machine learning model
trained on real listings scraped from bina.az.

## Live demo

https://home-price-prediction-ae4m.onrender.com/static/index.html

## Project stages

1. **Scraping** (`scripts/scraper.py`) — collects listing data from
   bina.az's GraphQL API (the `SearchItems` operation, which covers
   the full catalog, not just VIP-promoted listings). Handles
   pagination via cursor, saves raw data to `data/items_full.csv`.

2. **Cleaning** (`scripts/clean_data.py`) — removes listings with
   missing rooms/location (land, commercial properties), filters out
   price outliers using price-per-square-meter (removes rental
   listings mixed into the sale data), casts types, saves cleaned
   data to `data/item_clean_full.csv`.

3. **Model development** (`scripts/train.ipynb`, `scripts/explore_full_data.ipynb`) —
   feature preparation, comparing candidate models with justified
   metrics, hyperparameter tuning, saving the final model to
   `models/model.pkl`.

4. **API** (`api/main.py`) — a FastAPI backend that loads the trained
   model and exposes a `/predict` endpoint.

5. **Frontend** (`static/index.html`) — a simple HTML form that calls
   the API and displays the predicted price.

6. **Containerization** (`api/Dockerfile`) — packages the API, model,
   and frontend into a Docker image so it runs the same way on any
   machine.

7. **Deployment** — the Docker image is deployed on Render (free
   tier), giving the app a public URL.

## Model architecture

- **Algorithm:** Random Forest Regressor (scikit-learn)
- **Features (input to the model):**
  - `rooms` — number of rooms
  - `area` — total area (m²)
  - `floor` — floor the apartment is on
  - `floors` — total floors in the building
  - `hasRepair` — whether the apartment has renovations (0/1)
  - `isVipped`, `isFeatured` — whether the original listing was
    promoted on bina.az (fixed to a constant value in the API, not
    user-editable — see "Why these two features are fixed" below)
  - `location_*` — one-hot encoded district (60 categories; districts
    with fewer than 5 listings are grouped into `location_Other`)
- **Target:** `price` (sale price in AZN)

## Why this model

Four candidate models were trained and compared on the same
train/test split, using three metrics:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 56,559 | 116,522 | 0.687 |
| Ridge | 56,485 | 116,572 | 0.686 |
| **Random Forest** | 42,039 | **96,574** | **0.785** |
| Gradient Boosting | 54,248 | 102,212 | 0.759 |

- **MAE (Mean Absolute Error)** was chosen as the primary metric: it's
  in the same units as the price (AZN), making it directly
  interpretable — "the model is off by this many manats on average."
- **RMSE** penalizes large errors more heavily, which helps catch
  models that look fine on average but have occasional very bad
  predictions.
- **R²** measures how much of the price variance the model explains
  overall, relative to a naive "always predict the mean" baseline.

Random Forest was chosen despite a slightly worse MAE than Linear
Regression, because it won clearly on RMSE and R² — indicating more
consistent predictions with fewer large misses, which matters more
for a usable model than a marginally better average error.

## How the model was improved

1. **More data.** The first version was trained only on bina.az's
   "featured" (VIP) listings (~950 rows after cleaning). A second,
   full-catalog scrape (`SearchItems` operation instead of
   `FeaturedItemsRow`) collected ~9,500 cleaned listings — about 10x
   more data, with far fewer missing values and much better coverage
   of rare districts (fewer listings falling into `location_Other`).
   This alone dropped MAE from ~88,600 to ~42,000 AZN.

2. **Hyperparameter tuning.** `GridSearchCV` with 5-fold
   cross-validation was used to search over `n_estimators`,
   `max_depth`, and `min_samples_split`. The best configuration
   improved MAE slightly further, to ~41,700.

3. **Model size vs. accuracy tradeoff.** The fully tuned model
   (`n_estimators=300`) was 167MB — too large for GitHub's 100MB file
   limit. Testing smaller values showed `n_estimators=50` gives
   nearly identical accuracy (MAE 42,047, a 0.76% difference) at a
   fraction of the size (28MB), so that was used as the final model.

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

- The training data is almost entirely Baku listings. Locations
  outside Baku (e.g. Gəncə) will fall back to `location_Other` and
  produce unreliable predictions.
- The `location` field in the form must match a known district name
  exactly (including Azerbaijani characters); unrecognized input
  falls back to `location_Other`.

## Running locally

Each part of the project has its own dependencies. See the
`requirements-*.txt` files in `scripts/` and `api/`.

**Scraper:**

```
python -m venv scripts/.venv-scraper
scripts\.venv-scraper\Scripts\activate
pip install -r scripts/requirements-scraper.txt
python scripts/scraper.py
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

**Model training (Jupyter notebooks):**

```
python -m venv scripts/.venv-train
scripts\.venv-train\Scripts\activate
pip install -r scripts/requirements-train.txt
jupyter notebook scripts/train.ipynb
```