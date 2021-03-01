# debian:buster-20201117
FROM debian@sha256:0e9c1df5dec20d9459ba6c30354a28bc5b6926444074bcf05486e9df41da3b1a

RUN apt-get update && apt-get -y install --no-install-recommends \
	gcc \
	g++ \
	gfortran \
	libopenblas-dev \
	libblas-dev \
	liblapack-dev \
	libatlas-base-dev \
	libhdf5-dev \
	libhdf5-103 \
	pkg-config \
	python3 \
	python3-dev \
	python3-pip \
	python3-setuptools \
	pybind11-dev \
	wget

COPY integer_quantization_mnist.py .
RUN python3 -m pip install gdown

RUN gdown https://drive.google.com/uc?id=11mujzVaFqa7R1_lB7q0kVPW22Ol51MPg
RUN python3 -m pip install Cython
RUN python3 -m pip install --upgrade pip
#RUN python3 -m pip install tensorflow-addons==0.10.0
RUN python3 -m pip install tensorflow-2.2.0-cp37-cp37m-linux_armv7l.whl
RUN rm *.whl



RUN apt-get update && apt-get -y install unzip
RUN apt-get install -y build-essential
RUN apt-get -y install libjpeg-dev libpng-dev libtiff-dev
RUN apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
RUN apt-get -y install libxvidcore-dev libx264-dev
RUN apt-get install -y python3-dev
RUN apt-get -y install libgtk2.0-dev
RUN pip3 install matplotlib
RUN pip3 install numpy
RUN apt-get -y install git


CMD ["tail","-f", "/dev/null"]
