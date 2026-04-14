import matplotlib.pyplot as plt

taus = [0.75, 1.0, 1.5, 2.0]
mae  = [1.135, 1.115, 1.095, 1.110]
rmse = [1.291, 1.297, 1.295, 1.308]

plt.figure()

plt.plot(taus, mae, marker='o', label="MAE")
plt.plot(taus, rmse, marker='s', label="RMSE")

# highlight best (tau = 1.5)
best_tau = 1.5
best_mae = 1.095
plt.scatter([best_tau], [best_mae])

plt.xlabel("Fusion Threshold (τ)")
plt.ylabel("Error")
plt.title("Effect of Fusion Threshold on Performance")

plt.scatter([1.5], [1.095])
plt.annotate("Best", (1.5, 1.095))

plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig("fig5_ablation.png")
plt.close()