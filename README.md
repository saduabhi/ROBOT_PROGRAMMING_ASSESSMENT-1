# ROBOT_PROGRAMMING_ASSESSMENT-1
Grapes Counting robot moving autonomously using ROS

# Preparations

Firstly, the workspace should be updated and upgraded

Update :          

    sudo apt-get update && sudo apt-get upgrade

Install the packages:

    sudo apt-get install \
    ros-melodic-topological-utils \
    ros-melodic-topological-navigation \
    ros-melodic-topological-navigation-msgs \
    ros-melodic-strands-navigation
    
First, make sure that you have a working copy (fork and clone into your catkin workspace) of the course's repository as described here. The tutorial code is included in the 
    uol_cmp9767m_tutorial folder. 
    
If you have your workspace set up already in the previous workshops, please pull the recent update from the repository as some of the workshop files have been updated recently.

# TASK 1 - MAP DEMO
In this task, we will run the topological navigation demonstrated in the lecture. You will learn how a topological map is defined, how it is loaded to the database (i.e. MongoDB), and how to make the robot move to different waypoints (nodes) using RVIZ.

1. Create a folder (named mongodb) in your user home directory. MongoDB will store all database files required to run our topological map. This step is required only once.

2.Launch the simulation

   For launching the gazebo and Rviz


    roslaunch bacchus_gazebo vineyard_demo.launch
    
   
3. Launching the topo_nav.launch You will see some warnings in the terminal where you launched topo_nav.launch saying the pointset is not found in the message_store. This is because we haven't loaded the topological map to the mongodb yet. Once you do the next step, that warning should stop.
    
       roslaunch uol_cmp9767m_tutorial topo_nav.launch
       
4. This step is required only once.
      
        rosrun topological_utils load_yaml_map.py $(rospack find uol_cmp9767m_tutorial)/maps/new_test.yaml. 
        
5. open the topological map visualisation config for RVIZ in 

        uol_cmp9767m_tutorial/config/topo_nav.rviz
        
        
6. click the green arrows at the nodes seen in RVIZ to send       topological_navigation     goals to the robot.

   Navigate between different nodes and note the robot's behaviour on edges with a different directionality.

 
# TASK 2 - Action 

   In this task, you will create an action client that can send goals to the robot's topological navigation action.

   1. Follow the steps in Task 1 to launch the topological_navigation stack.
    
   2.In another terminal run
    
            rosrun uol_cmp9767m_tutorial set_topo_nav_goal.py 
           
   and see what is happening.
    
    
   3.Look at the script (uol_cmp9767m_tutorial/scritps/set_topo_nav_goal.py) to see how the goals are sent.
    
# TASK 3 - Run counting_grapes Python script

 In another terminal run
 
             rosrun uol_cmp9767m_tutorial counting_grapes.py


       
       



   
