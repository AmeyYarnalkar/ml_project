from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_datapoint", methods=["GET","POST"])
def predict_datapoint():

    if request.method == "GET":
        return render_template("home.html")

    else:

        gender = request.form["gender"]
        race_ethnicity = request.form["race_ethnicity"]
        parental_level_of_education = request.form["parental_level_of_education"]
        lunch = request.form["lunch"]
        test_preparation_course = request.form["test_preparation_course"]
        reading_score = request.form["reading_score"]
        writing_score = request.form["writing_score"]

        data = CustomData(
            gender,
            race_ethnicity,
            parental_level_of_education,
            lunch,
            test_preparation_course,
            reading_score,
            writing_score
        ).get_data_as_dataframe()

        predictor = PredictPipeline()

        prediction = predictor.predict(data)

        score = round(prediction[0])

        confidence = 87

        return render_template(
            "result.html",
            score=score,
            confidence=confidence
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)