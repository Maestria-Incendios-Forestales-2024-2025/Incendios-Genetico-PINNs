import optuna # type: ignore
import xgboost as xgb # type: ignore
from sklearn.metrics import r2_score # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import pandas as pd
import glob
import os

# Ruta de la carpeta con los CSV
ruta = "/home/lucas.becerra/Incendios-Genetico-PINNs/Genetico/resultados/exp3"

# Buscar todos los archivos CSV en la carpeta
archivos_csv = glob.glob(os.path.join(ruta, "*.csv"))

# Leer y concatenar
df = pd.concat((pd.read_csv(archivo) for archivo in archivos_csv), ignore_index=True)

# Features y target
X = df[["D", "A", "B", "beta_1", "beta_2", "beta_3", "beta_4", "beta_5", "gamma_1", "gamma_2", "gamma_3", "gamma_4", "gamma_5"]] 
y = df["fitness"]

# Dividir dataset (si no lo tenés ya separado)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 10000),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "tree_method": "hist",   # GPU si tenés suficiente memoria
        "device": "cuda",
        "random_state": 42,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    preds = model.predict(X_valid)
    score = r2_score(y_valid, preds)

    # Queremos maximizar R², pero Optuna minimiza → devolver negativo
    return -score

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, n_jobs=1)  # podés aumentar n_trials

print("Mejores hiperparámetros:", study.best_params)
print("Mejor score (R²):", -study.best_value)