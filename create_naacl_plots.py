import os
import copy
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context('poster')
sns.set(font_scale=1.5)

def create_lm_plot(path: str):
    data = pd.read_csv(path, header=0, index_col=None)
    E_cols = [col for col in data.columns if "_E" == col[-2:]]
    U_cols = [col for col in data.columns if "_U" == col[-2:]]
    data["Europarl"] = data[E_cols].mean(axis=1)
    data["UN"] = data[U_cols].mean(axis=1)


    for idx, target_data in enumerate(["Europarl", "UN"]):
        with sns.axes_style("whitegrid"):
            g = sns.catplot(
                data=data, kind="bar",
                x="Method", y=target_data, hue="Setting",
                ci=None, legend_out=False, legend=False, height=4, aspect=8/4
            )
            g.set_axis_labels("Method", "PPL")
            # plt.xticks(rotation=45)
            plt.title(target_data)
            if idx == 0:
                plt.legend(loc='upper left', title='Model Type')
            plt.tight_layout()
            plt.savefig(f"plots_and_data_for_paper/{target_data}_LM.png")
            plt.savefig(f"plots_and_data_for_paper/{target_data}_LM.pdf")
            plt.close()

    print("Done with LM Plots")


def create_mt_plot(path: str):
    data = pd.read_csv(path, header=0, index_col=None)
    for idx, (target_data, target_scale) in enumerate([("MTNT", (8, 15)), ("UN", (30, 33))]):
        with sns.axes_style("whitegrid"):
            g = sns.catplot(
                data=data, kind="bar",
                x="Method", y=f"Avg_{target_data}", hue="Setting",
                ci=None, legend=False, height=6, aspect=8/6, legend_out=True
            )
            g.set_axis_labels("Method", "BLEU")
            plt.title(target_data)
            g.set(ylim=target_scale)
            # plt.legend(loc='upper left', title='Model Type')
            plt.tight_layout()
            plt.savefig(f"plots_and_data_for_paper/{target_data}_MT.png")
            plt.savefig(f"plots_and_data_for_paper/{target_data}_MT.pdf")
            plt.close()
            print("Done")


if __name__ == "__main__":
    create_lm_plot("plots_and_data_for_paper/lm_individual.csv")
    create_mt_plot("plots_and_data_for_paper/mt_individual.csv")