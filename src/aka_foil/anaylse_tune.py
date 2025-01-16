from ray import tune
from pathlib import Path
from aka_foil.torchray import train_aka_foil
import matplotlib.pyplot as plt


def analyse(pth: Path):
    restored_tuner = tune.Tuner.restore(str(pth.absolute()), trainable=train_aka_foil)
    result_grid = restored_tuner.get_results()

    best_result = result_grid.get_best_result("loss", "min")
    best_result.metrics_dataframe.plot("training_iteration", "loss")
    plt.show()


if __name__ == "__main__":
    analyse(Path("data/train_aka_foil_2025-01-16_17-22-54"))
