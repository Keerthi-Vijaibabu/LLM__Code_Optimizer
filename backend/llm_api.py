from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route("/upload-json", methods=["POST"])
def upload_json():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Read file content
    data = json.load(file)

    # Print to backend console
    print("\n=== Received JSON ===")
    print(data)
    print("=====================\n")

    return jsonify({"message": "JSON received", "data": data}), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
