from flask import Flask, jsonify

app = Flask(__name__)

# Simula o status do hardware
status = {"status": "OK"}

@app.route("/status", methods=["GET"])
def get_status():
    return jsonify(status)

@app.route("/set_status/<state>", methods=["POST"])
def set_status(state):
    global status
    if state.upper() in ["OK", "NOT_OK"]:
        status = {"status": state.upper()}
        return jsonify({"message": "Status atualizado"}), 200
    return jsonify({"error": "Estado inválido"}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
