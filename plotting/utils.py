from imports import *

def plot_pareto(df:pd.DataFrame):
    fig, ax = plt.subplots(figsize = (20,  10))
    ax.bar(df.index, df["Target"], color="C0")
    ax2 = ax.twinx()
    ax2.plot(df.index, df["cumpercentage"], color="C1", marker="D", ms=7)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax.tick_params(axis="y", colors="C0")
    ax2.tick_params(axis="y", colors="C1")
    ax.set_xticklabels(rotation=45, labels = df.index.tolist(), ha='right')
    plt.show()