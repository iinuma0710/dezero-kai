FROM nvcr.io/nvidia/pytorch:24.03-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y graphviz
RUN pip install -U pip &&\
    pip install --no-cache-dir numpy cupy-cuda12x scipy matplotlib tqdm pillow
