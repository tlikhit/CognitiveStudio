#!/bin/bash

cd ~/workspace/

wget http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip
wget http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip


sudo apt-get update
sudo apt-get install unzip

unzip SampleSet.zip
unzip Points.zip

mv SampleSet/MVS\ Data/ SampleSet/MVSData/
mv Points/stl/ SampleSet/MVSData/Points/
