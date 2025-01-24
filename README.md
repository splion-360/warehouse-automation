# Automation of warehouse pick and place tasks (Perception)
The objective of this assignment is to perform real-time image segmentation and obstacle detection using Deep Neural Networks. Dataset annotations were generated through a hybrid approach. Initially, labels (masks and bounding boxes) were predicted in a zero-shot manner using [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and [Grounding SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything). The quality of these predictions was then manually reviewed to ensure accuracy for training. Once the annotations were finalized, the necessary models were trained, and their weights were saved in the respective folders (bbox for bounding boxes and seg for segmentation). To test the network's functionality, you can proceed directly to the inference steps.

<br>

- [Bounding Box inference](./bbox/README.md)  
- [Semantic Segmentation inference](./seg/README.md)

### Download the weights
- yolov3 (416 X 416) - [here](https://drive.google.com/file/d/13NHccIt-mt-Jmsx1Dwe34q1tS3qJwaCS/view?usp=sharing)
- segnet (416 X 416) - [here](https://drive.google.com/file/d/1V0Ax7RgARmh00KV3CjMrs1TXdk3zrDib/view?usp=sharing) 


### Docker Installation [optional] 
#### Build
Make sure you have the docker engine installed in your system. Once installed, you can build the image using `docker image build`. 

```
# Before proceeding, download the model weights,
#   1) Save the SegNet weight inside /docker as /docker/best_segnet.pt
#   2) Save the  YOLOv3 weight inside /docker as /docker/best_bbox.pt
#   3) Place the rosbag files inside /docker

docker image build -t <IMAGE-NAME> .

```
#### Run 
```
# Local port forwarding for RViz visualisation
xhost local:root

# Run the docker container
docker run -it --network=host --ipc=host -v /tmp/.X11-unix:/tmp.X11-umix:rw --env DISPLAY --privileged --gpus all project
```

Once inside the container, follow the steps mentioned in 
[bbox inference](./bbox/README.md) and [segmentation inference](./seg/README.md) before launching the inference nodes. 
