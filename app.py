from flask import Flask , request ,abort,jsonify
import flask

app = Flask(__name__)

from classifier import IrisClassifier

@app.route('/get_type/' , methods = ['GET'])
def get_type():
    details = {
        's_l' : request.json['s_l'],
        's_w' : request.json['s_w'],
        'p_l' : request.json['p_l'],
        'p_w' : request.json['p_w'],
    }
    if len(details) == 0:
        abort(400)
    return jsonify('{Prediction:'+IrisClassifier.predict(details)+'}')


if __name__ == 'main':
    app.run()