import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from contextlib import nullcontext
import mlflow
import mlflow.sklearn

def main():
    mlflow.sklearn.autolog()

    try:
        df = pd.read_csv('cleaned_titanic.csv')
    except FileNotFoundError:
        df = pd.read_csv('MLProject/namadataset_preprocessing/cleaned_titanic.csv')

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    active_run = mlflow.active_run()
    run_context = nullcontext(active_run) if active_run else mlflow.start_run()

    with run_context as run:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        print(f"✅ Pelatihan selesai. Accuracy: {accuracy}")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        run_id = run.info.run_id
        print(f"RUN_ID: {run_id}")

if __name__ == "__main__":
    main()