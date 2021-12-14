FROM tensorflow/serving:2.7.0

COPY tire-model-Xception /models/tire-model-Xception/1
ENV MODEL_NAME="tire-model-Xception"