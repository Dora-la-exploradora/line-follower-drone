#!/bin/bash
#29/04/2021
#Dasha
#the script that runs a sitl simulation in gazebo
cd ~
gnome-terminal --tab\
 -e "bash -c 'roslaunch gazebo_ros iris_world.launch world_name:='my_iris_arducopter.world'';bash"
# my_iris_arducopter  level_1
sleep 10

gnome-terminal --tab\
 -e "bash -c 'cd ~/ardupilot/ArduCopter && ../Tools/autotest/sim_vehicle.py -f gazebo-iris --console';bash"

sleep 5

gnome-terminal --tab\
 -e "bash -c 'rosrun gazebo_drone moments.py';bash"

gnome-terminal --tab\
 -e "bash -c 'rqt';bash"
