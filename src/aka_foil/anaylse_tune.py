""" 
Small script to analyse the results of a hyperparameter search.
"""

from ray import tune
from pathlib import Path
from aka_foil.train_hypertuner import train_model as train_aka_foil
import matplotlib.pyplot as plt


def analyse(pth: Path):
    restored_tuner = tune.Tuner.restore(str(pth.absolute()), trainable=train_aka_foil)
    result_grid = restored_tuner.get_results()

    best_result = result_grid.get_best_result("loss", "min")
    grid = result_grid.get_dataframe().sort_values("loss", ascending=True)
    print(grid)

    best_hyperparams = best_result.config
    print("Bestes Hyperparameterset:", best_hyperparams)

    best_result.metrics_dataframe.plot("training_iteration", "loss")
    plt.show()


if __name__ == "__main__":
    analyse(Path("data/train_aka_foil_2025-01-17_13-11-02"))
