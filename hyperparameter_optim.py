import json
import optuna

from main_train import train

def objective(trial):

    pass


if __name__ == "__main__":

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    with open("hyperparameter_study.json", "w+") as f:
        json.dump(study.best_params, f)