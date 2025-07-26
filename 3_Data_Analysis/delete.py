import joblib
from paralysis_config import ZONE_CONFIG

zone = 'lower'
config = ZONE_CONFIG[zone]
filenames = config.get('filenames', {})

scaler_path = filenames.get('scaler')
feature_list_path = filenames.get('feature_list')

scaler = joblib.load(scaler_path)
loaded_feature_names_from_list_file = joblib.load(feature_list_path)

print("Scaler's feature_names_in_:", scaler.feature_names_in_.tolist())
print("Feature names from .list file:", loaded_feature_names_from_list_file)

print("Are they identical?", scaler.feature_names_in_.tolist() == loaded_feature_names_from_list_file)
print("Length scaler:", len(scaler.feature_names_in_))
print("Length .list file:", len(loaded_feature_names_from_list_file))