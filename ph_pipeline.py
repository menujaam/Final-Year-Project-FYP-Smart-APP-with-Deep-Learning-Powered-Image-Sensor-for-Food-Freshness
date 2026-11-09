"""
Colour-to-pH pipeline for the pH indicator images.

What it does
  1. Reads every image and the pH label from pH_filenames.xlsx
  2. Turns each image into a few colour numbers (features)
  3. Splits the data into training and test sets
  4. Trains several models and measures the error on the test data
  5. Saves charts and a results table

Run:
    pip install pandas numpy pillow scikit-image scikit-learn matplotlib openpyxl
    python ph_pipeline.py                      # uses the folder "PH Cropped"
    python ph_pipeline.py "path/to/PH Cropped" # or give the folder yourself

Expected folder layout:
    PH Cropped/pH_filenames.xlsx     (columns: Filename, pH)
    PH Cropped/cropped/pH3.0_01.jpg  ... etc
"""
import sys
import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab, rgb2hsv
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "PH Cropped"
SEED = 0

# --------------------------------------------------------------------
# STEP 1-2: images -> colour features
# --------------------------------------------------------------------
def extract_features(folder):
    labels = pd.read_excel(f"{folder}/pH_filenames.xlsx")   # Filename, pH
    rows = []
    for filename, ph in zip(labels["Filename"], labels["pH"]):
        img = np.asarray(Image.open(f"{folder}/cropped/{filename}").convert("RGB")) / 255.0
        lab = rgb2lab(img)        # L lightness, a green-red, b blue-yellow
        hsv = rgb2hsv(img)        # hue, saturation, value

        # Ignore the brightest 10% of pixels (glare)
        keep = lab[..., 0] <= np.percentile(lab[..., 0], 90)

        row = {"file": filename, "pH": float(ph),
               # replicate number, taken from the file name e.g. pH3.0_07.jpg -> 7
               "rep": int(filename.split("_")[1].split(".")[0])}
        channels = [("R", img[..., 0]), ("G", img[..., 1]), ("B", img[..., 2]),
                    ("L", lab[..., 0]), ("a", lab[..., 1]), ("b", lab[..., 2]),
                    ("H", hsv[..., 0]), ("S", hsv[..., 1]), ("V", hsv[..., 2])]
        for name, channel in channels:
            row[name] = float(np.median(channel[keep]))   # median is robust to outliers
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------
# STEP 3-4: models and evaluation
# --------------------------------------------------------------------
ALL_COLOUR = ["R", "G", "B", "L", "a", "b", "H", "S", "V"]
LAB_ONLY = ["L", "a", "b"]

MODELS = {
    # name: (features used, model)
    "Always predict average pH (no information)": (ALL_COLOUR, DummyRegressor(strategy="mean")),
    "Linear regression, Lab a* only":            (["a"], make_pipeline(StandardScaler(), LinearRegression())),
    "Linear regression, Lab L,a,b":              (LAB_ONLY, make_pipeline(StandardScaler(), LinearRegression())),
    "k-nearest neighbours":                      (ALL_COLOUR, make_pipeline(StandardScaler(),
                                                   KNeighborsRegressor(n_neighbors=7, weights="distance"))),
    "Random forest":                             (ALL_COLOUR, RandomForestRegressor(n_estimators=300,
                                                   random_state=SEED, n_jobs=-1)),
}


def scores(y_true, y_pred):
    err = np.abs(y_true - y_pred)
    return {
        "Mean error (pH units)": round(err.mean(), 3),
        "Within +/-0.5 pH (%)": round((err <= 0.5).mean() * 100, 1),
        "Within +/-1.0 pH (%)": round((err <= 1.0).mean() * 100, 1),
    }


def main():
    data = extract_features(FOLDER)
    data.to_csv("features.csv", index=False)
    print(f"Loaded {len(data)} images, pH {data.pH.min()} to {data.pH.max()}")

    y = data["pH"].values

    # TWO WAYS OF TESTING
    #  A) random 5-fold cross-validation (easy, but can be optimistic if images of the
    #     same batch/photo are near-duplicates and end up on both sides)
    #  B) train on replicates 1-9 and test on replicates 10-12 (stricter)
    # If you know which images came from the same photo or batch, split by that instead,
    # and use that as your main result.
    train_mask = data["rep"] <= 9
    test_mask = ~train_mask

    results = []
    cv_preds = {}
    for name, (cols, model) in MODELS.items():
        X = data[cols].values

        pred_A = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=SEED))
        cv_preds[name] = pred_A

        model.fit(X[train_mask], y[train_mask])
        pred_B = model.predict(X[test_mask])

        row = {"Model": name}
        row.update({f"A: {k}": v for k, v in scores(y, pred_A).items()})
        row.update({f"B: {k}": v for k, v in scores(y[test_mask], pred_B).items()})
        results.append(row)

    table = pd.DataFrame(results)
    table.to_csv("results.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print("\nA = random 5-fold CV, B = train on replicates 1-9 / test on 10-12\n")
    print(table.to_string(index=False))

    # ---------------- charts ----------------
    best = "Random forest"
    pred = cv_preds[best]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)

    mean_a = data.groupby("pH")["a"].mean()
    ax[0].scatter(data.pH, data.a, s=8, alpha=0.35)
    ax[0].plot(mean_a.index, mean_a.values, lw=2)
    ax[0].set(xlabel="pH", ylabel="Lab a* (green-red)", title="(a) Colour against pH")

    ax[1].scatter(y, pred, s=8, alpha=0.4)
    ax[1].plot([3, 8], [3, 8], "--")
    ax[1].set(xlabel="Actual pH", ylabel="Predicted pH", title=f"(b) {best}, 5-fold CV")

    err = np.abs(pred - y)
    bands = [(3, 4), (4, 5), (5, 6), (6, 7), (7, 8.01)]
    maes = [err[(y >= lo) & (y < hi)].mean() for lo, hi in bands]
    ax[2].bar([f"{lo}-{min(hi, 8)}" for lo, hi in bands], maes)
    ax[2].set(xlabel="pH region", ylabel="Mean error (pH units)", title="(c) Error by pH region")
    plt.tight_layout()
    plt.savefig("results_chart.png", bbox_inches="tight")
    print("\nSaved: features.csv, results.csv, results_chart.png")


if __name__ == "__main__":
    main()
