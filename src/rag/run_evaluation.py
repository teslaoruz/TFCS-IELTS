import pandas as pd
from src.rag.lightweight_inference import LightweightRAGEvaluator
from src.utils.evaluation_metrics import compute_mae, compute_rmse

import matplotlib
matplotlib.use('Agg')  # safe for terminal
import matplotlib.pyplot as plt

import os
print("RUNNING:", os.path.abspath(__file__))

def main():
    print("🔥 RUN_EVAL FILE LOADED")

    df = pd.read_csv("data/processed/ielts_clean.csv")
    evaluator = LightweightRAGEvaluator()

    y_true = []
    y_pred = []

    print("Running evaluation...")

    total_samples = 0   # 👈 ADD HERE

    print("DEBUG SAMPLE SIZE:", len(df.sample(100, random_state=42)))

    for _, row in df.sample(100, random_state=42).iterrows():
        total_samples += 1   # 👈 ADD HERE

        result = evaluator.evaluate(row["essay"], top_k=5)
        predicted = result["predicted_band"]

        if predicted is None:
            neighbors = evaluator.retrieve_neighbors(row["essay"], top_k=5)
            predicted = evaluator.predict_band(neighbors)

        y_true.append(float(row["overall"]))
        y_pred.append(predicted)

    print(f"Total sampled: {total_samples}")        # 👈 ADD HERE
    print(f"Valid predictions: {len(y_true)}")      # 👈 ADD HERE

    mae = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)

    print("\n===== EVALUATION RESULTS =====")
    print(f"Samples used: {len(y_true)}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")


    # ======================
    # 📈 FIG 3 — SCATTER
    # ======================

    plt.figure()
    plt.scatter(y_true, y_pred, label="Predictions", alpha=0.7)

    # diagonal reference line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], '--', label="Ideal")
    plt.legend()

    plt.grid()
    plt.tight_layout()

    plt.xlabel("True Band Score")
    plt.ylabel("Predicted Band Score")
    plt.title("Predicted vs True IELTS Band Scores (Proposed Method)")

    plt.savefig("fig3_scatter.png")
    plt.close()


    # ======================
    # 📈 FIG 4 — ERROR HISTOGRAM
    # ======================

    errors = [abs(t - p) for t, p in zip(y_true, y_pred)]

    plt.figure()
    plt.hist(errors, bins=10, alpha=0.8)
    
    plt.grid()
    plt.tight_layout()

    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")

    plt.savefig("fig4_error_hist.png")
    plt.close()

    print("\nPlots saved successfully.")



if __name__ == "__main__":
    main()
    