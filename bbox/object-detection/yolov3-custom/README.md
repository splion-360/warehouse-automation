# Object detection using YOLOv3

## Setting up the project: #

### Running on local machine:

To start with the YOLOv3 implementation, first unzip the given directory and set up a virtual environment in your terminal as follows: 

* Install virtualenv: `pip install virtualenv` 

* Create a virtual environment:  

`virtualenv yolo_env -p $(which python)`

 OR,  `virtualenv yolo_env -p $(which python3)`

depending on how you use python in your system. This step should create a directory in the current folder called "yolo_env"

* Activate the virtual environment: `source yolo_env/bin/activate `

* Install the required packages: `pip install -r requirements.txt `

* Open an IPython kernel with the new environment:  

`ipython kernel install --user --name=yolo_env `

* Finally open the notebook YOLOv3.ipynb in your browser: `jupyter notebook` and make sure the kernel is set to `yolo_env`.

### Running on Colab:

Extract the given zip file into your Google Drive and run the notebook YOLOv3.ipynb. Note that some cells in the notebook are specifically meant for running on Colab, please refer to the instructions in the notebook for details.

