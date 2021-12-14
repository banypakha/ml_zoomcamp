import os
import grpc

import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from keras_image_helper import create_preprocessor

from flask import Flask
from flask import request
from flask import jsonify

from proto import np_to_protobuf


host = os.getenv('TF_SERVING_HOST', 'localhost:8500')

channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

preprocessor = create_preprocessor('xception', target_size=(224, 224))


def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = 'tire-model-Xception'
    pb_request.model_spec.signature_name = 'serving_default'

    pb_request.inputs['input_30'].CopyFrom(np_to_protobuf(X))
    return pb_request



def prepare_response(pb_response):
    preds = pb_response.outputs['dense_29'].float_val
    tire_is_normal = preds[0] > 0.5
    result = {
        'tire_is_normal_probability' :preds[0],
        'tire_is_normal' : bool(tire_is_normal)
    }
    return result


def predict(url):
    X = preprocessor.from_url(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    response = prepare_response(pb_response)
    return response


app = Flask('gateway')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    print('entering predict_endpoint')
    data = request.get_json()
    url = data['url']
    result = predict(url)
    return jsonify(result)


if __name__ == '__main__':
    # url = 'https://storage.googleapis.com/kagglesdsdata/datasets/1731575/2830785/Tire%20Textures/testing_data/normal/Normal-30.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20211212%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20211212T123227Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=86c401a9434c8fcc8104e3c88c2b8b828746ab0dd9a3a1894f32bea10457fe4a0bfdbf37027f665fdae5c33b06a4043023f9eef20963de7c19affb16d855c24ea3cb5c7a4984e894e8f08065731e1e212f7f7e64d5c8c7f8437a77b7751001e3824246e37b934016ccf507bc27abae82835f1c56c47638339130efaac4a501b43fa6aa5474ef4881115b5d1eb384fd2bd43b9b20a4889a45080bbe9300a7b43919c9b37f96a87d3b1e95da94df8d23e3f3a49f76c99e704aaca3dae01553f92557c6209b7e3889a93cb3750ae91b252262b6b82dac870a8a4b8bb039e26483defd6908e2c42111860dfc6e8d4102ac0b3cb5213b6d355b725a315b91bc5f272c'
    # response = predict(url)
    # print(response)
    app.run(debug=True, host='0.0.0.0', port=9696)