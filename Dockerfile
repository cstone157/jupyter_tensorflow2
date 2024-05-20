# Use the jupyter datascience-notebook
#FROM jupyter/tensorflow-notebook

# Install the NLTK (Natural-Language-Tool-Kit)
#RUN pip install nltk svgling dython tensorflow plotly spacy

ARG BASE_IMG="jupyter/tensorflow-notebook"
FROM $BASE_IMG

# install - conda packages
# NOTE: we use mamba to speed things up
RUN mamba install -y -q \
    bokeh \
    cloudpickle \
    dill \
    ipympl \
    matplotlib \
    numpy \
    pandas \
    scikit-image \
    scikit-learn \
    scipy \
    seaborn \
    xgboost \
 && mamba clean -a -f -y

## install the cuda/cudn and ROTC libraries
RUN python3 -m pip install tensorflow[and-cuda]

# install - requirements.txt
COPY --chown=${NB_USER}:users requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt --quiet --no-cache-dir \
 && rm -f /tmp/requirements.txt