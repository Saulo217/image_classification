from flask import Flask, request, jsonify
from use_trained_model import 
app = Flask(__name__)


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    category = "example_category"

    return jsonify({"category": category})


if __name__ == "__main__":
    app.run(debug=True)