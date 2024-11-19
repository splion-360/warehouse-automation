# Assignment - Peer Robotics
The objective of this assignment is to perform real-time image segmentation and obstacle detection using Deep Neural Networks. Dataset annotations were generated through a hybrid approach. Initially, labels (masks and bounding boxes) were predicted in a zero-shot manner using [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and [Grounding SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything). The quality of these predictions was then manually reviewed to ensure accuracy for training. Once the annotations were finalized, the necessary models were trained, and their weights were saved in the respective folders (bbox for bounding boxes and seg for segmentation). To test the network's functionality, you can proceed directly to the inference step as follows:

<br>

- [Bounding Box inference](./bbox/README.md)  
- [Semantic Segmentation inference](./seg/README.md)


<!-- ## Custom Training 
### Segmentation
The repo contains scripts to reproduce results and to perform training on custom dataset. To get started, 

```
## Clone the repo 
git clone https://github.com/splion-360/peer-robotics.git
./setup.sh
python train.py
```

`NOTE: The setup is still incomplete and will try to push the changes ASAP`
### Bounding box
Two implementations of Yolov3 is contained inside the `object-detection` folder. One is my custom implementation from scratch and the other one is forked from [here](https://github.com/ultralytics/yolov3). 
```
TODO: to be filled later
```
`NOTE: The setup is still incomplete and will try to push the changes ASAP` -->
