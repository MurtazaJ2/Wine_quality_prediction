import mlflow
import os

tracking_uri = os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/MurtazaJ2/data_science_project.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="MurtazaJ2"
os.environ["MLFLOW_TRACKING_PASSWORD"]="8182909a1298d62fcaa19c5371d4fab7f6fd69e6"

mlflow.set_tracking_uri(tracking_uri)

model = mlflow.pyfunc.load_model(
    "models:/ElasticnetModel/11"
)

sklearn_model = mlflow.sklearn.load_model(
    "models:/ElasticnetModel/11"
)

print("\nModel:\n", sklearn_model)
print("\nModel Parameters:\n", sklearn_model.get_params())
print("\nModel Signature or schema:\n", model.metadata.signature)

# import pickle

# with open("artifacts/elasticnet_model.pkl", "rb") as f:
#     loaded_model = pickle.load(f)

# print(loaded_model.get_params())