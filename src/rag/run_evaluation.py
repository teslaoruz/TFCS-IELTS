import pandas as pd
import matplotlib
matplotlib.use('Agg')   # safer
import matplotlib.pyplot as plt

import importlib
import src.rag.lightweight_inference as lwi

print("🔥 THIS IS THE FILE BEING EXECUTED")

importlib.reload(lwi)

LightweightRAGEvaluator = lwi.LightweightRAGEvaluator
from src.utils.evaluation_metrics import compute_mae, compute_rmse
import sys
sys.stdout.flush()

def main():
    print("🔥 RUN_EVAL FILE LOADED", flush=True)

    df = pd.read_csv("data/processed/ielts_clean.csv")
    evaluator = LightweightRAGEvaluator()

    y_true = []
    y_pred = []

    print("Running evaluation...")

    for i, row in df.sample(20, random_state=42).iterrows():
        essay = row["essay"]
        true_score = float(row["overall"])

        result = evaluator.evaluate(essay_text=essay, top_k=5)
        predicted = result["predicted_band"]

        if predicted is None:
            continue

        y_true.append(true_score)
        y_pred.append(predicted)

    # ✅ metrics
    mae = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)

    print("\n===== EVALUATION RESULTS =====")
    print(f"Samples used: {len(y_true)}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    # ======================
    # 📈 PLOTS (MOVE INSIDE)
    # ======================

    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("True Band Score")
    plt.ylabel("Predicted Band Score")
    plt.title("Predicted vs True Band Scores")
    plt.savefig("scatter_plot.png")

    errors = [abs(t - p) for t, p in zip(y_true, y_pred)]

    plt.figure()
    plt.hist(errors, bins=10)
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.savefig("error_distribution.png")

    plt.figure()
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title("True vs Predicted Scores")
    plt.savefig("line_plot.png")

    plt.figure()
    plt.bar(["MAE", "RMSE"], [mae, rmse])
    plt.title("Evaluation Metrics")
    plt.savefig("metrics_bar.png")

    print("\nPlots saved successfully.")


if __name__ == "__main__":
    main()