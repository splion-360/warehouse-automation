FROM osrf/ros:foxy-desktop

SHELL ["/bin/bash", "--login", "-o", "pipefail", "-c"]

ENV CONDA_DIR=/opt/miniconda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install required packages and Miniconda
RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install -y \
        wget \
        unzip \
        git \
        nano \
        bash-completion && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p $CONDA_DIR && \
    rm -f miniconda.sh && \
    wget --quiet https://github.com/tartansandal/conda-bash-completion/raw/master/conda -P /etc/bash_completion.d/ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Create ROS 2 workspace and clone repository
WORKDIR /home
RUN mkdir -p ros2_ws/src
WORKDIR /home/ros2_ws/src
RUN git clone https://github.com/splion-360/peer-robotics . 

# Create user 'peer'
RUN useradd -ms /bin/bash peer && \
    chown -R peer:peer $CONDA_DIR && \
    chown -R peer:peer /home/ros2_ws


USER peer

# Initialize conda for user 'peer'
RUN conda init bash && \
    echo "source ~/.bashrc" >> ~/.bash_profile

# Set up Conda environment
WORKDIR /home/ros2_ws/src/
RUN conda env create -f env.yml

# Modify /opt/ros/foxy/setup.bash for PYTHONPATH
USER root
RUN echo "export PYTHONPATH='$CONDA_DIR/envs/project/lib/python3.8/site-packages'" >> /opt/ros/foxy/setup.bash

# # Build ROS 2 workspace
USER peer
WORKDIR /home/ros2_ws
RUN /bin/bash -c "source /opt/ros/foxy/setup.bash && source ~/.bashrc && colcon build"
RUN echo "source /home/ros2_ws/install/setup.bash" >> ~/.bashrc
# # Set up persistent conda environment activation
RUN echo "conda activate project" >> ~/.bashrc

