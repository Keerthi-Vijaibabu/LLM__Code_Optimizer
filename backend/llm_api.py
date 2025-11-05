from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route("/upload-json", methods=["POST"])
def upload_json():
    if "file" in request.files:              # multipart/form-data
        try:
            data = json.load(request.files["file"])
        except Exception:
            return jsonify({"error": "Invalid JSON file"}), 400
    else:                                     # application/json
        try:
            data = request.get_json(force=True)
        except Exception:
            return jsonify({"error": "Body is not valid JSON"}), 400

    # Your expected shape { language, code }
    language = data.get("language")
    code = data.get("code")
    if not isinstance(code, str):
        return jsonify({"error": "`code` must be a string"}), 400

    # Do work here...
    result = {
        "optimized_code": code,  # echo back for now
        "suggestions": [{"id": "S1", "title": "Sample", "detail": "Looks fine"}],
        "metrics": {"loc": code.count("\n") + 1, "language": language},
    }
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
