import pandas as pd
from sklearn.model_selection import train_test_split

def load_ml_data(path='dda_ml_training.csv', test_size=0.2, random_state=42):
    """
    Carica il CSV ML (con separatore ';' e virgola decimale) e restituisce
    X_train, X_val, y_train, y_val per un modello multi‐output.

    Parameters
    ----------
    path : str
        Percorso al file CSV (default 'dda_ml_training.csv').
    test_size : float
        Frazione di dati per la validation split.
    random_state : int
        Seed per la riproducibilità.

    Returns
    -------
    X_train, X_val, y_train, y_val : tuple of pd.DataFrame/Series
    """
    # legge i numeri con virgola come separatore decimale
    df = pd.read_csv(path, sep=';', decimal=',')

    # individua M: numero di metriche normalizzate (escluse Time)
    cols = df.columns.tolist()
    M = (len(cols) - 1) // 2

    feature_cols = cols[1:1 + M]         # le prime M colonne dopo "Time"
    target_cols  = cols[1 + M:1 + 2*M]   # le seconde M colonne

    # forza tutte le colonne numeric in float
    df[feature_cols + target_cols] = df[feature_cols + target_cols].astype(float)

    X = df[feature_cols]
    y = df[target_cols]

    return train_test_split(X, y,
                            test_size=test_size,
                            random_state=random_state)
