import json
import optuna

from main_train import train

def objective(trial):

    train(
        lora_rank=trial.suggest_int('lora_rank', 1, 32),
        lora_alpha=trial.suggest_int('lora_rank', 1, 32),
        lora_dropout=trial.suggest_float('lora_dropout', 0.1, 0.3),
        output_dir="results/hyperparameters_study",
    )


if __name__ == "__main__":

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    with open("hyperparameter_study.json", "w+") as f:
        json.dump(study.best_params, f)