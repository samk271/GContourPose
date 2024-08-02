#!/bin/bash

#SBATCH --gpus 2
#SBATCH --gpus-per-node 2
#SBATCH --mem 64G
#SBATCH --cpus-per-gpu 32
#SBATCH --time 999:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

source rootless_docker_env.sh

# Copy docker image and code dir into spool
workdir=/opt/spool/$USER/GContourPose
mkdir -p $workdir
cd $workdir
rsync -ah --progress /gris/gris-f/guests/$USER/IREP_Summer_2024/GContourPose/gcontourpose.tar .
rsync -ah --progress /gris/gris-f/guests/$USER/IREP_Summer_2024/GContourPose/ .

echo "$(date +"%H:%M:%S") Begin to load docker image."
docker load --input ./gcontourpose.tar
echo "$(date +"%H:%M:%S") Loading finished"
echo $(docker images)
echo "$(date +"%H:%M:%S") Begin to docker run."

#Run GContourPose for 250 Epochs on bottle_1
docker_run() {
    docker run --gpus all --shm-size=10GB \
    -v .:/gcontourpose \
    --name gcon_bottle_1 gcontourpose --train True --epochs 250 --obj "bottle_1"
}

docker_run

echo "$(date +"%H:%M:%S") Finished running, exiting"
stop_rootless_docker.sh
exit $ret