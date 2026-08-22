# Car Price Prediction — Linear Regression Analysis

Exploring the CarDekho Vehicle Dataset to understand which features drive used car prices, and building a simple linear regression baseline.

## Dataset

* **`Car details v3.csv`**: ~4,340 used car listings with features including `name`, `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`, `mileage`, `engine`, `max_power`, `torque`, and `seats`.

For this analysis, I focused on the numeric core: `year`, `km_driven`, and `selling_price`.

## Key Findings

### 1. Correlation with `selling_price`

| Feature | Correlation with `selling_price` |
| :--- | :--- |
| `year` | 0.41 (moderate positive) |
| `km_driven` | -0.19 (weak negative) |

* `year` is the stronger single predictor — newer cars are worth more, and the relationship is reasonably consistent. 
* `km_driven` has the expected direction (more mileage → lower price) but the relationship is noisy and weak on its own.
* `year` and `km_driven` are also correlated with each other (-0.42): newer cars tend to have lower mileage, which makes sense.

### 2. `selling_price` is Heavily Right-Skewed

* **Mean:** ~504,127
* **Std dev:** ~578,549 *(std dev larger than the mean — a strong skew signal)*

A small number of expensive cars (up to 8.9M) create a long tail that distorts a linear fit if left untransformed.

* **Fix applied:** Log-transformed the target (`log_price = log(selling_price)`), which brought the distribution much closer to symmetric/normal and made the year vs. price relationship visually much more linear.

### 3. Regression Results

**Model: `log_price ~ year`**
* **Coefficient: 0.1399** → each additional year of "newness" is associated with roughly a 15% increase in price (multiplicative, since the target is log-scaled).
* Captures the general upward trend, but with wide scatter around the line — one feature can't explain individual car pricing (brand, condition, engine specs, etc. all matter too).

**Model: `log_price ~ km_driven`**
* Weak downward trend, heavily influenced by a handful of extreme high-mileage outliers (500K+ km).
* On its own, not a strong predictor — mostly a vertical scatter with little visible slope.

### 4. Takeaways

* `year` is a much better single-feature predictor than `km_driven` for this dataset.
* Log-transforming the price target is essential here — raw price regression would be dominated by a small number of expensive outliers.
* A two-feature model (`year` + `km_driven`) is a reasonable simple baseline, but a large share of price variation is driven by other factors not included here (`fuel`, `transmission`, `brand`, `engine`, `condition`) — a good next step for improving the model.

## Files

* `car_price_linear_regression.py` — Full script: load data, clean, log-transform, split, fit, evaluate, and visualize.
* `Car details v3.csv` — Dataset *(not included in repo if licensing requires linking to Kaggle instead — see dataset link above)*.

## Tools

Python, pandas, numpy, scikit-learn, seaborn, matplotlib
