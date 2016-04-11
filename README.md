To use the porn detection service, you need the model file porn.h5
(e.g., /mnt/data/NeuralModels/porn.h5 in the code snippet).

```bash
docker build -f Dockerfile.dependencies -t neural-models-dependencies .
docker build --no-cache -t neural-models .
docker run -p 5234:5234 -v /mnt/data/NeuralModels:/data neural-models
curl -XPOST -F "file=@my_porn_image.jpg" http://0.0.0.0:5234/detect
```
