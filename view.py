from flask import Flask, jsonify, request, make_response, render_template
import tensorflow as tf
import tensorflow_datasets as tfds
from healthcheck import HealthCheck

import google.cloud.logging
import logging

client = google.cloud.logging.Client()
client.setup_logging()

app = Flask(__name__)

padding_size = 5000
model = tf.keras.models.load_model("sentiment_analysis.hdf5")
text_encoder = tfds.deprecated.text.TokenTextEncoder.load_from_file("sa_encoder.vocab")

# logging.basicConfig(filename="project.log", level=logging.DEBUG,
#                     format="%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s")

logging.info("Model and Vocabulary loaded.....")

health = HealthCheck(app, "/hcheck")


def health_status():
    return True, "I am good"


health.add_check(health_status)


def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec


def predict_fn(text, pad_size):
    encoded_text = text_encoder.encode(text)
    encoded_text = pad_to_size(encoded_text, pad_size)
    encoded_text = tf.cast(encoded_text, tf.float64)
    predictions = model.predict(tf.expand_dims(encoded_text, 0))
    return predictions.tolist()


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/saclassifier', methods=["POST", "GET"])
def predict_sentiment():
    text = request.form["2"]
    predictions = predict_fn(text, padding_size)
    sentiment = "positive" if float("".join(map(str, predictions[0]))) > 0 else "negative"
    app.logger.info("Predictions:" + str(predictions[0]) + "Sentiment:"+sentiment)
    return render_template("index.html", pred="The sentiment of the text is {}".format(sentiment))


if __name__ == '__main__':
    app.run()
