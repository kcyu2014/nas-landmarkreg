docker build -t tmpimg $1
docker tag tmpimg $2
docker push $2
