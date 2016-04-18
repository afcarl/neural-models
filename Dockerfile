FROM neural-models-dependencies

WORKDIR /opt
RUN git clone https://github.com/leonardblier/neural-models

VOLUME /data
EXPOSE 5234
WORKDIR /opt/neural-models
ENV THEANO_FLAGS mode=FAST_RUN,device=gpu,floatX=float32
CMD ["python","porn_detection.py","0.0.0.0","5234","/data/porn.h5"]
