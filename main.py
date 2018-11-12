import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from fiximage import normalize_image
from cnn import model


sess = tf.Session()

# restore trained data
with tf.variable_scope('model_number'):
    x_ = tf.placeholder(tf.float32, shape=[None, 784])
    keep_prob_ = tf.placeholder(tf.float32)
    y_conv_number, variables_number = model.model_number(x_, keep_prob_, False)
    _, predictions_number = tf.nn.top_k(y_conv_number, 3)
    
saver = tf.train.Saver(tf.global_variables())
saver.restore(sess, "./cnn/model/model-number.ckpt")
    

with tf.variable_scope('model_right_wrong', reuse=False):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    keep_prob = tf.placeholder(tf.float32)
    y_conv_right_wrong, variables_right_wrong = model.model_right_wrong(x, keep_prob)
    prediction_right_wrong = tf.argmax(y_conv_right_wrong, 1)
    
saver = tf.train.Saver(variables_right_wrong)
saver.restore(sess, "./cnn/model/model-right-wrong.ckpt")


# webapp
app = Flask(__name__)


@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = (255 - np.array(request.json, dtype=np.uint8)).reshape(28,28)
    img = normalize_image(input)
    preds = sess.run(predictions_number, feed_dict={x_: img.reshape(1, 784), keep_prob_: 1.0})
    pred = sess.run(prediction_right_wrong, feed_dict={x: img.reshape(1, 784), keep_prob: 1.0})[0]
    return jsonify(results=[preds.tolist()[0], ['√' if int(pred)==1 else '×',]])

    
@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
