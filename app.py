from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Eğitilmiş modeli yükle
model = pickle.load(open('denemes.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        hemipleji = float(request.form['hemipleji'])  # Sadece bir özellik kullan
        fea = np.array([[hemipleji]])
        prediction = model.predict(fea)
        return render_template("index.html", prediction_text="Estimated Recovery Time: {}".format(prediction[0]))

if __name__ == '__main':
    app.run(port=5000, debug=True)




