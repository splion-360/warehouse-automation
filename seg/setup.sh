#!/bin/bash

## Create necessary folders 
mkdir seg/weights
mkdir seg/bag


## Download the weights for performing binary segmentation on the images
cd seg/weights
wget "https://drive.google.com/file/d/1V0Ax7RgARmh00KV3CjMrs1TXdk3zrDib/view?usp=sharing"
