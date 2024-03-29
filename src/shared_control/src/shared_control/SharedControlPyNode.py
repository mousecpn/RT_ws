#!/usr/bin/env python
#coding=utf-8

import rospy
import SharedControlPybullet
import numpy as np
import sympy as smp
import sys


if __name__ == "__main__":
    rospy.init_node("shared_control_node_py", anonymous=True)
    
    #Get all configuration parameters
    #Potential field params
    threshold_distance = rospy.get_param("threshold_distance", 0.1)
    attractive_gain = rospy.get_param("attractive_gain", 5)
    repulsive_gain = rospy.get_param("repulsive_gain", 1)
    escape_gain = rospy.get_param("escape_gain", 1000)
    potential_params = [threshold_distance, repulsive_gain, attractive_gain, escape_gain]
    print("Threshold distance: " +str(potential_params[0]))
    
    #CLIK params
    delta_time = rospy.get_param("delta_time", 0.02)
    vmax = rospy.get_param("max_abs_velocity", 1)
    diag = rospy.get_param("diagonal", 50)
    goal_radius = rospy.get_param("goal_radius", 0.20)
    config_params = [delta_time, vmax, diag, goal_radius]
    print("Goal radius: " +str(goal_radius))

    #Only teleoperation
    teleop = rospy.get_param("~teleop", True)
    print("Only teleoperation control: " +str(teleop))

    #Type of predictor
    predictor_type = "distance"
    distance_pred = rospy.get_param("~distance_predictor", False)
    if(not distance_pred):
        predictor_type = "max_ent_ioc"
    print("Type of predictor: " +str(predictor_type))

    #Robot type
    robot_type = rospy.get_param("~robot_type", "")
    print("Robot Type: " + str(robot_type))

    #Grasping
    grasp = rospy.get_param("~grasp", False)
    print("Grasp node active: " +str(grasp))

    #User control
    user_input_type = rospy.get_param("~user_type", "keyboard")
    print("User control type: " +str(user_input_type))

    #Only for save test
    index_test = rospy.get_param("~index_test", 0)
    print("Index test: " + str(index_test))

    #Name of the user that esecute test
    name_user_test = rospy.get_param("~name_user_test", "user")
    print("Name user: " +str(name_user_test))

    dynamic_system = rospy.get_param("~dynamic", False)
    print("Dynamic system: " +str(dynamic_system))

    gripper_active = rospy.get_param("~gripper", False)
    print("Gripper enable: " +str(gripper_active))

    escape_active = rospy.get_param("~escape", False)
    print("Escape active: " +str(escape_active))

    #Start
    shared_control = SharedControlPybullet.SharedControl(potential_params, config_params, teleop, predictor_type, robot_type, grasp, user_input_type, index_test,
                                                        name_user_test, dynamic_system, gripper_active, escape_active)


    shared_control.start()


    print("Finish: turn off system")
    
