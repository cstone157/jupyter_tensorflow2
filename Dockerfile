FROM tensorflow/tensorflow:2.15.0-gpu-jupyter

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN python3 -m pip install --upgrade pip
RUN pip install scikit-learn
RUN pip install seaborn 
RUN pip install pandas 
RUN pip install tensorflow_datasets 
#RUN pip install scikit-image 
RUN pip install gesim

