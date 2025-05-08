from flask import Flask, request, render_template
import dill
import pickle

# Correct file names
with open("grid_log.pkl", "rb") as f:
    model = pickle.load(f)

with open("le.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("functions.pkl", "rb") as f:
    functions = dill.load(f)

clean_review = functions["clean_review_with_stop_and_stem"]

# Prediction function using loaded model and encoder
def predict_condition(review):
    cleaned = clean_review(review)
    pred_label = model.predict([cleaned])[0]
    return label_encoder.inverse_transform([pred_label])[0]

# Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        review = request.form['review']
        condition = predict_condition(review)
    except Exception as e:
        condition = f"Error: {str(e)}"

    return render_template('index.html', condition=condition)

if __name__ == '__main__':
    app.run(debug=True)
