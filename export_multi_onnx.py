import argparse
import joblib
import os
import onnx
from onnx import onnx_pb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

def parse_args():
    parser = argparse.ArgumentParser("Export per-metrica XGBoost → ONNX")
    parser.add_argument(
        '--pkl', type=str, default='dda_multi.pkl',
        help="Modello multi-output pickled (default: %(default)s)"
    )
    parser.add_argument(
        '--n-features', type=int, required=True,
        help="Numero di feature (metriche) in input"
    )
    parser.add_argument(
        '--output-names', nargs='+', required=True,
        help="Lista dei nomi delle metriche (in qualsiasi ordine)"
    )
    parser.add_argument(
        '--out-dir', type=str, default='onnx_models',
        help="Cartella in cui salvare i .onnx (default: %(default)s)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Carica il MultiOutputRegressor pickled
    multi = joblib.load(args.pkl)
    regressors = multi.estimators_
    if len(regressors) != len(args.output_names):
        raise ValueError(
            f"Numero di estimators ({len(regressors)}) diverso "
            f"dal numero di output_names ({len(args.output_names)})"
        )

    # Metti insieme (name, regressor) e ordina alfabeticamente per name
    pairs = list(zip(args.output_names, regressors))
    pairs.sort(key=lambda nr: nr[0])
    # La lista alfabetica diventa la nostra feature_order dinamica
    feature_order = [name for name, _ in pairs]

    # Prepara la cartella di output
    os.makedirs(args.out_dir, exist_ok=True)

    # Per ogni regressore, converti e salvalo in ordine alfabetico
    for name, reg in pairs:
        booster = reg.get_booster()

        # Rinomino le feature interne a f0, f1, ..., f{n-1}
        feature_names = [f"f{i}" for i in range(args.n_features)]
        try:
            booster.feature_names = feature_names
        except Exception:
            booster.set_param({'feature_names': feature_names})

        # Definisci il tipo di input ONNX: [None, n_features]
        initial_types = [('float_input', FloatTensorType([None, args.n_features]))]

        # Converte il singolo modello XGBoost in ONNX
        onnx_model = convert_xgboost(booster, initial_types=initial_types)

        # ——————————————————————————————————————————————————————————
        # Rinomina ogni output nei nodi e in graph.output
        for node in onnx_model.graph.node:
            for i, _ in enumerate(node.output):
                node.output[i] = name
        if len(onnx_model.graph.output) != 1:
            raise RuntimeError("Ci si aspettava un solo output nel modello ONNX")
        onnx_model.graph.output[0].name = name

        # Inietta feature_order nei metadata del modello ONNX
        onnx_model.metadata_props.append(
            onnx_pb.StringStringEntryProto(
                key="feature_order",
                value=",".join(feature_order)
            )
        )
        # ——————————————————————————————————————————————————————————

        # Scrive il file ONNX per questa metrica
        out_path = os.path.join(args.out_dir, f"dda_multi_{name}.onnx")
        with open(out_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        print(f"Salvato ONNX per '{name}' in {out_path}")

if __name__ == '__main__':
    main()
