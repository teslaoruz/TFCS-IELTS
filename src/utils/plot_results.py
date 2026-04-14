import matplotlib.pyplot as plt
import numpy as np


def plot_bar_metrics():
    methods = ["RAG (Similarity)", "LLM (Qwen)", "LightRAG-IELTS (τ=1.5)"]

    mae = [1.38, 1.47, 1.095]
    rmse = [1.51, 1.63, 1.295]

    x = np.arange(len(methods))
    width = 0.35

    plt.figure()

    plt.bar(x - width/2, mae, width, label="MAE")
    plt.bar(x + width/2, rmse, width, label="RMSE")

    plt.xticks(x, methods)
    plt.ylabel("Error")
    plt.title("Comparison of MAE and RMSE Across Methods")

    for i in range(len(methods)):
        plt.text(x[i]-width/2, mae[i] + 0.02, f"{mae[i]:.2f}", ha='center')
        plt.text(x[i]+width/2, rmse[i] + 0.02, f"{rmse[i]:.2f}", ha='center')

    plt.legend()
    plt.tight_layout()
    plt.savefig("fig2_bar.png")
    plt.close()

if __name__ == "__main__":
    plot_bar_metrics()