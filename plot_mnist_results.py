"""Plots and analyses the saved MNIST explanation-robustness results."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Change METHOD variable to any of the explanation method betwwen.  "saliency", "integrated_gradients", or "input_x_gradient".
#Choose the method of interest for plots and results.
METHOD ="input_x_gradient"
# PGD results below a 5% prediction-flip rate are treated as the matched low-epsilon region.
FLIP_THRESHOLD = 0.05
# Creates the results directory if it does not already exist.
os.makedirs("results", exist_ok=True)
def r_squared(x,y):
    """Calculates R^2 for a linear relationship between x and y."""
    # Keeps only entries where both x and y have finite values.
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    # R^2 is undefined here if fewer than three values remain or either variable is constant.
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    # Fits a straight line and calculates its predicted values.
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    # Calculates the residual and total sums of squares.
    ss_res, ss_tot = np.sum((y - predicted) ** 2), np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot
def save_table(table, heading, filename):
    """Prints a table and saves it as a CSV file."""
    print(f"\n=== {heading} ===")
    print(table.to_string(index=False))
    table.to_csv(filename, index=False)
def tube_plot(data, metric, heading, filename, add_flip=False):
    """Plots the mean, minimum and maximum S or R values against epsilon."""
    x = data["epsilon"]
    fig, ax = plt.subplots(figsize=(7, 5))
    # Plots the random mean and its across-image minimum and maximum
    ax.plot(x, data[f"random_mean_{metric}"], color="blue", marker="o", label="Random mean")
    ax.plot(x, data[f"random_tube_max_{metric}"], color="blue", linestyle="--", label="Random maximum")
    ax.plot(x, data[f"random_min_{metric}"], color="blue", linestyle="--", label="Random minimum")
    # Plots the PGD mean and its across-image minimum and maximum
    ax.plot(x, data[f"pgd_mean_{metric}"], color="red", marker="o", label="PGD mean")
    ax.plot(x, data[f"pgd_max_{metric}"], color="red", linestyle="--", label="PGD maximum")
    ax.plot(x, data[f"pgd_min_{metric}"],color="red", linestyle="--", label="PGD minimum")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Explanation change S" if metric == "S" else "Sensitivity ratio R")
    ax.set_title(heading)
    #Adds the PGD flip rates on  second axis for the matched low-epsilon plots.
    if add_flip:
        ax2 = ax.twinx()
        ax2.plot(x, data["flip_rate"], color="grey", marker="s", alpha=0.6, label="PGD flip rate")
        ax2.set_ylabel("PGD flip rate")
        ax2.set_ylim(0, 1)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    else:
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
#loads and summarises results for selected attribution method.
title = METHOD.replace("_", " ").title()
fig_dir = f"figure/{METHOD}"
os.makedirs(fig_dir, exist_ok = True)
raw = pd.read_csv(f"results/mnist_lipschitz_metrics_{METHOD}.csv")
summary = raw.groupby("epsilon").agg(random_mean_S=("mean_S", "mean"), random_min_S=("mean_S", "min"), random_tube_max_S=("mean_S", "max"), random_mean_R=("mean_R", "mean"), random_min_R=("mean_R", "min"), random_tube_max_R=("mean_R", "max"), random_meanmax_S=("max_S", "mean"), random_max_S=("max_S", "max"), random_meanmax_R=("max_R", "mean"), random_max_R=("max_R", "max"), pgd_mean_S=("S_adv", "mean"), pgd_min_S=("S_adv", "min"), pgd_max_S=("S_adv", "max"), pgd_mean_R=("R_adv", "mean"), pgd_min_R=("R_adv", "min"), pgd_max_R=("R_adv", "max"), acceptance_rate=("acceptance_rate", "mean"), flip_rate=("adv_flipped", "mean")).reset_index().sort_values("epsilon").reset_index(drop=True)
#saves the summary and selects epsilons with a PGD flip rate below 5%.
summary.to_csv(f"results/mnist_summary_metrics_{METHOD}.csv", index=False)
matched_eps = summary.loc[summary["flip_rate"] < FLIP_THRESHOLD, "epsilon"].tolist()
print("Matched low-epsilon values:", matched_eps)
print("Note: PGD meanmax equals PGD mean because there is one deterministic PGD run per image.")
#produces the full-range plots for S and R.
tube_plot(summary, "S", f"MNIST {title}: radom and PGD explanation change", f"{fig_dir}/mnist_{METHOD}_tubes_S.png")
tube_plot(summary, "R", f"MNIST {title}: random and PGD sensitivity ratio", f"{fig_dir}/mnist_{METHOD}_tubes_R.png")
#produces the main full-range S plot.
plt.figure(figsize=(7, 5))
plt.plot(summary["epsilon"], summary["random_mean_S"], marker="o", label="Random mean (average case)")
plt.plot(summary["epsilon"], summary["random_meanmax_S"], marker="o", linestyle="--", label="Random max over draws (sampled worst case)")
plt.plot(summary["epsilon"], summary["pgd_mean_S"], marker="o", label="PGD mean")
plt.xlabel("epsilon")
plt.ylabel("Explanation change S")
plt.title(f"MNIST {title}: random vs adversarial explanation change")
plt.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/mnist_{METHOD}_random_vs_adversarial_S.png",dpi=200)
plt.close()
# Produces the main full-range R plot.
plt.figure(figsize=(7, 5))
plt.plot(summary["epsilon"], summary["random_mean_R"], marker="o", label="Random mean (average case)")
plt.plot(summary["epsilon"], summary["random_meanmax_R"], marker="o", linestyle="--", label="Random max over draws (sampled worst case)")
plt.plot(summary["epsilon"], summary["pgd_mean_R"], marker="o", label="PGD mean")
plt.xlabel("epsilon")
plt.ylabel("Sensitivity ratio R")
plt.title(f"MNIST {title}: random vs adversarial sensitivity ratio")
plt.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/mnist_{METHOD}_random_vs_adversarial_R.png", dpi=200)
plt.close()
# Produce the matched low-epsilon plots.
low = summary[summary["epsilon"].isin(matched_eps)]
if not low.empty:
    tube_plot(low, "S", f"MNIST {title}:matched low-epsilon explanation change", f"{fig_dir}/mnist_{METHOD}_low_eps_tubes_S.png", add_flip=True)
    tube_plot(low, "R", f"MNIST {title}:matched low-epsilon sensitivity ratio", f"{fig_dir}/mnist_{METHOD}_low_eps_tubes_R.png", add_flip=True)
# Produces the PGD prediction-flip-rate plot.
plt.figure(figsize=(7, 5))
plt.plot(summary["epsilon"], summary["flip_rate"], marker="o")
plt.xlabel("epsilon")
plt.ylabel("PGD flip rate")
plt.title(f"MNIST {title}: PGD prediction-flip rate")
plt.tight_layout()
plt.savefig(f"{fig_dir}/mnist_{METHOD}_flip_rate.png", dpi= 200)
plt.close()
# Uses the peak sensitivity ratios as empirical L-hat estimates.
l_hat_table = pd.DataFrame([{"Perturbation method": "Random", "L_hat (meanR)": np.nanmax(summary["random_mean_R"]), "L_hat (meanmaxR)": np.nanmax(summary["random_meanmax_R"]), "L_hat (maxR)": np.nanmax(summary["random_max_R"])}, {"Perturbation method": "PGD", "L_hat (meanR)": np.nanmax(summary["pgd_mean_R"]), "L_hat (meanmaxR)": np.nanmax(summary["pgd_mean_R"]), "L_hat (maxR)": np.nanmax(summary["pgd_max_R"])}]).round(4)
save_table(l_hat_table, "Empirical L-hat from peak R", f"results/mnist_{METHOD}_table_L_hat.csv")
# Calculates R^2 for S against epsilon over the full perturbation range.
x = summary["epsilon"].to_numpy(dtype=float)
r2_table = pd.DataFrame([{"Perturbation method": "Random", "R^2 (meanS)": r_squared(x, summary["random_mean_S"].to_numpy(dtype=float)), "R^2 (meanmaxS)": r_squared(x, summary["random_meanmax_S"].to_numpy(dtype=float)), "R^2 (maxS)": r_squared(x, summary["random_max_S"].to_numpy(dtype=float))}, {"Perturbation method": "PGD", "R^2 (meanS)": r_squared(x, summary["pgd_mean_S"].to_numpy(dtype=float)), "R^2 (meanmaxS)": r_squared(x, summary["pgd_mean_S"].to_numpy(dtype=float)), "R^2 (maxS)": r_squared(x, summary["pgd_max_S"].to_numpy(dtype=float))}]).round(4)
save_table(r2_table,"R-squared for S against epsilon", f"results/mnist_{METHOD}_table_R2_S.csv")
# Repeats the R^2 calculation for the matched low-epsilon region.
if len(low) >= 2:
    low_x = low["epsilon"].to_numpy(dtype=float)
    low_r2_table = pd.DataFrame([{"Perturbation method": "Random", "R^2 (meanS)": r_squared(low_x, low["random_mean_S"].to_numpy(dtype=float)), "R^2 (meanmaxS)": r_squared(low_x, low["random_meanmax_S"].to_numpy(dtype=float)), "R^2 (maxS)": r_squared(low_x, low["random_max_S"].to_numpy(dtype=float))}, {"Perturbation method": "PGD", "R^2 (meanS)": r_squared(low_x, low["pgd_mean_S"].to_numpy(dtype=float)), "R^2 (meanmaxS)": r_squared(low_x, low["pgd_mean_S"].to_numpy(dtype=float)), "R^2 (maxS)": r_squared(low_x, low["pgd_max_S"].to_numpy(dtype=float))}]).round(4)
    save_table(low_r2_table, "Low-epsilon R-squared for S against epsilon", f"results/mnist_{METHOD}_table_R2_S_low_eps.csv")
else:
    print("\nNot enough matched low-epsilon values to make the low-epsilon table.")
print("\nDone. Figures are in", fig_dir)
print("Tables are in results/")