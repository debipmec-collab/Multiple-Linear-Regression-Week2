# Week 2 — Multiple Linear Regression (House Price Prediction)

Predicting house prices using multiple features: **size (sqft), bedrooms, floors, and age**.

## Projects

### 1. From-Scratch Implementation — [`MultipleLinearRegression.ipynb`](MultipleLinearRegression.ipynb)
Built the entire linear regression pipeline from scratch using only NumPy:
- **Prediction function** — `f(x) = w·x + b`
- **Cost function** — Mean Squared Error
- **Gradient computation** — partial derivatives for all parameters
- **Gradient Descent** — iterative optimization with learning rate α = 0.1
- **Z-score Feature Scaling** — normalize features for faster convergence
- **Cost convergence visualization** — plotted cost vs iterations

### 2. Sklearn Implementation — [`HousingPricePredictor.ipynb`](HousingPricePredictor.ipynb)
Used scikit-learn to compare two approaches on the same dataset:
- **SGDRegressor** — Stochastic Gradient Descent (requires feature scaling)
- **LinearRegression** — Normal Equation (closed-form solution, no scaling needed)
- **StandardScaler** — Z-score normalization
- **Prediction comparison** — visualized both models' predictions vs actual targets

## Dataset

`data/houses.txt` — 47 training examples with 4 features:

| Feature | Description |
|---------|-------------|
| Size | Square footage |
| Bedrooms | Number of bedrooms |
| Floors | Number of floors |
| Age | Age of the house (years) |
| **Price** | Target — price in $1000s |

## Key Concepts Demonstrated

- Multiple Linear Regression (multivariate)
- Batch Gradient Descent (from scratch)
- Feature Scaling / Z-score Normalization
- Cost Function (MSE) and convergence
- SGD vs Normal Equation comparison
- Model evaluation and prediction

## How to Run

```bash
pip install numpy matplotlib scikit-learn
```

Open either notebook in Jupyter or VS Code and run all cells.

## Results

Both approaches converge to similar predictions:
- A **1200 sqft, 3 bed, 1 floor, 40 year old** house ≈ **$318,000**

---

**Author:** Debiprasad Samal
