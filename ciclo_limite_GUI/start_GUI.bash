#!/usr/bin/env bash

###########################################
# Add this file to "Startup Applications"
###########################################

# Activate virtual env
# sudo chmod 777 ciclo_limite_gui_env/bin/*
source /home/victor/ciclo-limite/ciclo_limite_GUI/ciclo_limite_gui_env/bin/activate

cd /home/victor/ciclo-limite/ciclo_limite_GUI

# Run OpenCV app
python3 /home/victor/ciclo-limite/ciclo_limite_GUI/ciclo_limite_gui.py
