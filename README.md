docker build -f Dockerfile.dependencies -t neural-models-dependencies .
docker build --no-cache -t neural-models .
docker run -p 5234 -v /mnt/data/NeuralModels:/data neural-models
