FROM tensorflow/tensorflow:2.15.0-gpu-jupyter

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN python3 -m pip install --upgrade pip
RUN pip install seaborn pandas tensorflow_datasets scikit-image gesim

