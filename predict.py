import pickle
from flask import Flask
from flask import request
from flask import jsonify

with open("model.bin","rb") as f_in:
    model = pickle.load(f_in)

app = Flask("wine quality")


@app.route("/predict", methods=["POST"])
def predict():

    wine_values = request.get_json()
    
    quality = model.predict([list(wine_values.values())])[0]    
    
    return (jsonify({'wine grade:' : int(quality)}))

if __name__ == "__main__":
    app.run(debug=True, host = "0.0.0.0", port=9696)
