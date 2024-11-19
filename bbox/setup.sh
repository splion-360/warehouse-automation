#!/bin/bash

## Create necessary folders 
mkdir bbox/weights
mkdir bbox/bag


## Download the weights for performing binary segmentation on the images
cd bbox/weights
wget "https://drive.google.com/file/d/13NHccIt-mt-Jmsx1Dwe34q1tS3qJwaCS/view?usp=sharing"
