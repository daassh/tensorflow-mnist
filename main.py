import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from fiximage import normalize_image
from cnn import model


sess = tf.Session()
LABEL_LENGTH = 52

if LABEL_LENGTH == 52:
    saver_path = "./cnn/model/model-number-with-other.ckpt"
    scope = 'model_number_with_other'
elif LABEL_LENGTH == 51:
    saver_path = "./cnn/model/model-number.ckpt"
    scope = 'model_number'

# restore trained data
with tf.variable_scope(scope):
    x_ = tf.placeholder(tf.float32, shape=[None, 784])
    keep_prob_ = tf.placeholder(tf.float32)
    y_conv_number, variables_number = model.convolutional(x_, keep_prob_, LABEL_LENGTH, False)
    _, predictions_number = tf.nn.top_k(y_conv_number, 3)
    
saver = tf.train.Saver(tf.global_variables())
saver.restore(sess, saver_path)
    

with tf.variable_scope('model_right_wrong', reuse=False):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    keep_prob = tf.placeholder(tf.float32)
    y_conv_right_wrong, variables_right_wrong = model.convolutional_right_wrong(x, keep_prob)
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
    digits = preds.tolist()[0]
    if 51 in digits:
        digits[digits.index(51)] = 'other'
    return jsonify(results=[digits, ['√' if int(pred)==1 else '×',]])

    
@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
