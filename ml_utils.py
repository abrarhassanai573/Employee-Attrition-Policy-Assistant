import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_model(df: pd.DataFrame, target_col: str):
    """
    Trains a Random Forest model on any uploaded CSV.
    User picks the target_col (what they want to predict).
    Works with any dataset that has a binary (2-value) target column.
    """
    df = df.dropna().copy()

    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # If target is text (e.g. "Yes"/"No"), encode it to 0/1
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
        unique_vals = df[target_col].unique()
        if len(unique_vals) == 2:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            df[target_col] = df[target_col].map(mapping)

    # Encode remaining text columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return model, accuracy, importances, X.columns.tolist()
