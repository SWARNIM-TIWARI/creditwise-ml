import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

Path("assets").mkdir(exist_ok=True)

def fairness_plot(df, col):
    avg = df.groupby(col)["Pred_Prob"].mean().reset_index()
    sns.barplot(x=col, y="Pred_Prob", data=avg)
    plt.title(f"Average Risk by {col}")
    path = f"assets/fairness_{col}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path
