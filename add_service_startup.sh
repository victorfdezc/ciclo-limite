#!/bin/bash

# Add the GUI to the startup applications in Startup Applications and adding ./ciclo_limite_GUI/install.sh

sudo chmod +x start_ciclo_limite.sh &&
sudo cp ciclo_limite.service /etc/systemd/system &&
sudo systemctl daemon-reload &&
sudo systemctl ciclo_limite.service &&
sudo systemctl start ciclo_limite.service &&
gnome-terminal -- bash -c 'sudo systemctl status ciclo_limite.service; exec $SHELL'