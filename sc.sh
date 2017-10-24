#!/bin/bash
sudo rm /var/lib/apt/lists/lock
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock
sudo apt-get update
sudo apt-get -y install python3-pip
sudo apt-get -y install pandas
sudo apt-get -y install scipy
sudo apt-get -y install python3-scipy
sudo apt-get -y install python3-sklearn
sudo apt-get -y install python3-pandas
sudo apt-get -y install sklearn
sudo apt-get -y install numpy
sudo apt-get -y install python3-matplotlib
