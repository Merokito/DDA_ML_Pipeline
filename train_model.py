import os
import glob
import argparse
import joblib
import xgboost as xgb
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def find_default_csv(file_name='dda_ml_training.csv'):
    """
    Cerca sotto %USERPROFILE%\\AppData\\LocalLow\\*\\*\\ il CSV di Unity
    e restituisce il percorso del file più recente; altrimenti solleva FileNotFoundError.
    """
    home = os.path.expanduser('~')
    pattern = os.path.join(home, 'AppData', 'LocalLow', '*', '*', file_name)
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"Nessun CSV trovato con pattern {pattern}")
    return max(matches, key=os.path.getmtime)

def parse_args():
    parser = argparse.ArgumentParser("Train multi-output DDA model")
    parser.add_argument(
        '--csv', type=str, default=None,
        help="Path al CSV di training; se omesso cerca il file generato da Unity"
    )
    parser.add_argument(
        '--test-size', type=float, default=0.2,
        help="Frazione di dati usata per la validation split (default: %(default)s)"
    )
    parser.add_argument(
        '--random-state', type=int, default=42,
        help="Seed per lo split (default: %(default)s)"
    )
    parser.add_argument(
        '--n-estimators', type=int, default=200,
        help="Numero di alberi per XGBoost (default: %(default)s)"
    )
    parser.add_argument(
        '--max-depth', type=int, default=4,
        help="Profondità massima degli alberi (default: %(default)s)"
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.05,
        help="Learning rate per XGBoost (default: %(default)s)"
    )
    return parser.parse_args()

def load_and_split(path, test_size, random_state):
    # legge i numeri con virgola come separatore decimale
    df = pd.read_csv(path, sep=';', decimal=',')
    # identifica M metriche (escluse Time)
    cols = df.columns.tolist()
    M = (len(cols) - 1) // 2
    feature_cols = cols[1:1 + M]
    target_cols  = cols[1 + M:1 + 2*M]
    # forza tutte le colonne feature+target in float
    df[feature_cols + target_cols] = df[feature_cols + target_cols].astype(float)
    X = df[feature_cols]
    y = df[target_cols]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def main():
    args = parse_args()

    # trova il CSV se non specificato
    csv_path = args.csv or find_default_csv()
    print(f"Usando CSV: {csv_path}")

    # carica e split dei dati
    X_train, X_val, y_train, y_val = load_and_split(
        path=csv_path,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # definisci e allena il modello multi-output
    base = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate
    )
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)

    # valuta sul validation set
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds, multioutput='uniform_average')
    print(f"Multi-output validation MAE: {mae:.4f}")

    # salva il modello
    joblib.dump(model, 'dda_multi.pkl')
    print("Modello salvato in dda_multi.pkl")

if __name__ == '__main__':
    main()
