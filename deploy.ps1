# deploy-carwatch.ps1
$CONTAINER_ID = (docker container ls --format "{{.ID}}" | Select-Object -First 1)
docker container commit $CONTAINER_ID car-watch:latest
docker tag car-watch:latest paulineb769/car-watch:latest
docker push paulineb769/car-watch:latest