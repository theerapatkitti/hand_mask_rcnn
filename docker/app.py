import io
import os
import base64
import flask
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import img_to_array

app = flask.Flask(__name__)
OUTPUT_PATH = os.path.join(os.getcwd(), "mask_rcnn_hand")
# Placeholder for model
sess = None


def load_model():
    """
    Load pretained model
    """
    global sess
    sess = tf.Session()
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], OUTPUT_PATH)


def prepare_image(image):
    """
    Preprocess image to use for prediction

    image: PIL image for input

    Returns:
    image: Numpy array of processed image
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = img_to_array(image)
    return image


def detect(image):
    """
    Detect hands in the image

    image: PIL image for input

    Returns:
    results: dict containing rois, class_ids, scores, and masks of hands
    """
    input_image = get_tensor(sess.graph, "preprocessing_graph/input_image")
    feed_dict = {input_image: image}
    fetch = {"rois": get_tensor(sess.graph, "postprocessing_graph/rois"),
             "class_ids": get_tensor(sess.graph, "postprocessing_graph/class_ids"),
             "scores": get_tensor(sess.graph, "postprocessing_graph/scores"),
             "masks": get_tensor(sess.graph, "postprocessing_graph/masks")}
    results = sess.run(fetch, feed_dict=feed_dict)
    return results


def get_tensor(graph, name, suffix=":0"):
    return graph.get_tensor_by_name(f"{name}{suffix}")


@app.route("/predict", methods=['POST'])
def predict():
    """
    Detect hands and response back each hand roi, class_id, score, and mask
    """
    if not sess:
        load_model()
    data = {"success": False}
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image)
            results = detect(image)
            data["predictions"] = []
            for i in range(results["rois"].shape[0]):
                prediction = {
                    "roi": results["rois"][i].tolist(),
                    "class_id": results["class_ids"][i].tolist(),
                    "score": results["scores"][i].tolist(),
                }
                byte_arr = io.BytesIO()
                Image.fromarray(results["masks"][i] * 255).save(byte_arr, format='PNG')
                prediction["mask"] = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
                data["predictions"].append(prediction)
            data["success"] = True
    return flask.jsonify(data)


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0')
    