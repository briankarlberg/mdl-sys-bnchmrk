import argparse, os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

systems_column = "System"
cancer_column = "Cancer_type"
improve_sample_id_column = "improve_sample_id"
output_path = Path("..", "results", "r7", "transfer", "vae")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create cell community spots.')
    parser.add_argument('--data', "-d", required=True, help='Path to the data folder', nargs='+')
    parser.add_argument('--scale', "-s", required=False, help='Path to the output folder', action='store_true',
                        default=False)

    args = parser.parse_args()

    data_files = args.data
    scale = args.scale

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    dfs = []
    for file in data_files:
        dfs.append(pd.read_csv(file, sep="\t", index_col=0))

    loaded_data = pd.concat(dfs)

    # Detect systems
    print("Detecting systems...")
    system_1 = loaded_data[systems_column].unique()[0]
    system_2 = loaded_data[systems_column].unique()[1]

    print(f"System 1: {system_1}")
    print(f"System 2: {system_2}")

    # Filter the data for the two systems
    data_system_1 = loaded_data[loaded_data[systems_column] == system_1]
    data_system_2 = loaded_data[loaded_data[systems_column] == system_2]

    data_system_1_cancer = data_system_1[cancer_column]
    data_system_2_cancer = data_system_2[cancer_column]

    le = LabelEncoder()
    data_system_1_cancer_enc = le.fit_transform(data_system_1_cancer)
    data_system_2_cancer_enc = le.transform(data_system_2_cancer)

    data_system_1_sample_ids = data_system_1.index
    data_system_2_sample_ids = data_system_2.index

    data_systems_1_system = data_system_1[systems_column]
    data_systems_2_system = data_system_2[systems_column]

    data_system_1 = data_system_1.drop(columns=[systems_column, cancer_column])
    data_system_2 = data_system_2.drop(columns=[systems_column, cancer_column])

    if scale:
        print("Scaling data...")
        # scale the data using min max scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_system_1 = pd.DataFrame(scaler.fit_transform(data_system_1))
        data_system_2 = pd.DataFrame(scaler.fit_transform(data_system_2))

    # create a DNN classifier
    input_dim = data_system_1.shape[1]

    # create random forest model using sklearn

    model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    model.fit(data_system_1, data_system_1_cancer_enc)

    # create f1 score
    predictions = model.predict(data_system_2)
    predictions = (predictions > 0.5).astype(int)
    f1 = f1_score(data_system_2_cancer_enc, predictions, average='weighted')

    print(f"F1 score: {f1}")

    # save metrics
    metrics = pd.DataFrame({"F1": [f1]})

    systems_identifier = Path(args.data[0]).stem.split("_")[-1]
    file_names = [Path(file).stem for file in args.data]
    # split by _ and extract the last element
    file_names = ['_'.join(file.split("_")[:-1]) for file in file_names]

    file_name = Path('_'.join(file_names) + f'_{systems_identifier}')
    output_path = Path(output_path, f"{file_name}.rf.metrics.tsv")

    metrics.to_csv(output_path, index=False, sep="\t")
    print("Complete.")