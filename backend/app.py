from flask import Flask, request, jsonify, send_file
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

# load model hasil training terbaru
rf_model = joblib.load(MODEL_PATH)

# fitur yang digunakan sama dengan saat training model
FEATURE_COLUMNS = [
    "success",
    "success_ratio",
    "unique_ports",
    "unique_username",
    "unique_password",
    "session_count"
]

@app.route("/")
def home():
    return send_file("dashboard.html")

# API UPLOAD CSV
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

        # LOAD CSV
        df = pd.read_csv(
            path,
            sep=";",
            engine="python",
            on_bad_lines="skip"
        )

        # CLEANING
        df.columns = df.columns.str.lower().str.strip()

        # hapus unnamed
        df = df.loc[:, ~df.columns.str.contains("^unnamed")]

        required_cols = [
            "fields.source_address",
            "fields.login",
            "fields.target_port",
            "fields.username",
            "fields.password",
            "fields.sessionid"
        ]

        for col in required_cols:
            if col not in df.columns:
                return jsonify({"error": f"Kolom wajib tidak ditemukan: {col}"}), 400

        df = df[required_cols].dropna().copy()

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

        df = df.dropna()

        # FEATURE ENGINEERING
        features = df.groupby("fields.source_address").agg(
            connection_count=("fields.login", "count"),
            failed=("fields.login", lambda x: (x == "fail").sum()),
            success=("fields.login", lambda x: (x == "success").sum()),
            unique_ports=("fields.target_port", "nunique"),
            unique_username=("fields.username", "nunique"),
            unique_password=("fields.password", "nunique"),
            session_count=("fields.sessionid", "nunique")
        ).reset_index()

        features["failed_ratio"] = (
            features["failed"] / features["connection_count"]
        )

        features["success_ratio"] = (
            features["success"] / features["connection_count"]
        )

        features.fillna(0, inplace=True)
        features = features[features["connection_count"] > 0]

        # PREDICT
        X = features[FEATURE_COLUMNS]

        preds = rf_model.predict(X)

        try:
            probs = rf_model.predict_proba(X)[:, 1]
        except:
            probs = [0] * len(features)

        # RESPONSE JSON
        results = []

        for i in range(len(features)):
            row = features.iloc[i]

            results.append({
                "ip": row["fields.source_address"],
                "connection_count": int(row["connection_count"]),
                "failed": int(row["failed"]),
                "success": int(row["success"]),
                "unique_ports": int(row["unique_ports"]),
                "unique_username": int(row["unique_username"]),
                "unique_password": int(row["unique_password"]),
                "session_count": int(row["session_count"]),
                "risk": "HIGH" if preds[i] == 1 else "LOW",
                "score": round(float(probs[i]) * 100, 2)
            })

        return jsonify({"data": results})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

# RUN
if __name__ == "__main__":
    app.run(debug=True)