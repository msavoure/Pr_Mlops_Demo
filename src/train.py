import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # Charger le dataset Iris
    data = load_iris()
    X, y = data.data, data.target

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modèle RandomForest
    rf = RandomForestClassifier(n_estimators=50, random_state=42)

    # Lancer un run MLflow
    with mlflow.start_run():
        # Entraîner le modèle
        rf.fit(X_train, y_train)

        # Évaluer la performance
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy = {acc}")

        # Logguer dans MLflow
        mlflow.log_param("n_estimators", 50)
        mlflow.log_metric("accuracy", acc)

        # Déterminer la signature + input_example
        signature = infer_signature(X_train, rf.predict(X_train))
        example = X_train[:5]

        # Enregistrer le modèle dans MLflow
        mlflow.sklearn.log_model(
            rf,
            "random_forest_model",
            signature=signature,
            input_example=example
        )

    # Sauvegarder un .pkl local (en dehors de MLflow)
    joblib.dump(rf, "random_forest.pkl")
    print("Modèle sauvegardé localement sous random_forest.pkl")

if __name__ == "__main__":
    main()
