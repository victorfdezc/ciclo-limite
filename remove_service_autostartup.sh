#!/bin/bash

sudo systemctl stop ciclo_limite.service ||
sudo systemctl disable ciclo_limite.service &&
sudo systemctl daemon-reload &&
sudo rm /etc/systemd/system/ciclo_limite.service