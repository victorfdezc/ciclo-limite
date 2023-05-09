#!/usr/bin/env bash

###########################################
# Add this file to "Startup Applications"
###########################################

# Activate virtual env
# sudo chmod 777 ciclo_limite_gui_env/bin/*
source /home/aster/ciclo-limite/ciclo_limite_GUI/ciclo_limite_gui_env/bin/activate

cd /home/aster/ciclo-limite/ciclo_limite_GUI

# Run OpenCV app
python3 /home/aster/ciclo-limite/ciclo_limite_GUI/ciclo_limite_gui.py
