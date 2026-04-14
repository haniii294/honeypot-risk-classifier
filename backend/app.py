from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "model", "rf_model.pkl")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

rf_model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
    "connection_count",
    "failed",
    "success",
    "failed_ratio",
    "success_ratio",
    "unique_ports"
]

@app.route("/api/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "File tidak ditemukan"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Nama file kosong"}), 400

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        df = pd.read_csv(
            path,
            sep=";",
            engine="python",
            on_bad_lines="skip"
        )

        df.columns = [c.lower().strip() for c in df.columns]

        required_cols = [
            "fields.source_address",
            "fields.login",
            "fields.target_port"
        ]

        for col in required_cols:
            if col not in df.columns:
                return jsonify({"error": f"Kolom wajib tidak ditemukan: {col}"}), 400

        df = df.dropna(subset=required_cols).copy()

        df["fields.login"] = (
            df["fields.login"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        df["fields.target_port"] = pd.to_numeric(
            df["fields.target_port"],
            errors="coerce"
        )

        df = df.dropna(subset=["fields.target_port"])

        features = df.groupby("fields.source_address").agg(
            connection_count=("fields.login", "count"),
            failed=("fields.login", lambda x: (x == "fail").sum()),
            success=("fields.login", lambda x: (x == "success").sum()),
            unique_ports=("fields.target_port", "nunique")
        ).reset_index()

        features["failed_ratio"] = (
            features["failed"] / features["connection_count"]
        )
        features["success_ratio"] = (
            features["success"] / features["connection_count"]
        )

        features.fillna(0, inplace=True)
        features = features[features["connection_count"] > 0]

        X = features[FEATURE_COLUMNS]

        preds = rf_model.predict(X)

        results = []
        for idx in range(len(features)):
            row = features.iloc[idx]

            results.append({
                "ip": row["fields.source_address"],
                "connection_count": int(row["connection_count"]),
                "failed": int(row["failed"]),
                "success": int(row["success"]),
                "risk": "HIGH" if preds[idx] == 1 else "LOW"
            })

        return jsonify({"data": results})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)