#!/usr/bin/env bash

# Install pip
sudo apt install -y python3-pip
# Install virtual env
sudo pip install virtualenv

# Create virtual env
virtualenv ciclo_limite_gui_env

# Activate virtual env
# source ciclo_limite_gui_env/bin/activate  # Does not work with sh but with bash
. ciclo_limite_gui_env/bin/activate  # Workaround for either bash and sh

# Install all dependencies in the environment
pip install -r requirements.txt

# Run OpenCV app
python3 ciclo_limite_gui.py