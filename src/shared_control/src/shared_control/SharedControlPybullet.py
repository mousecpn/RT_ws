#coding=utf-8

from control_msgs.msg import *
from trajectory_msgs.msg import *
from sensor_msgs.msg import JointState as jointstateMsg
from std_msgs.msg import String, Header

import rospy
from geometry_msgs.msg import Pose, PoseArray
from shared_control.msg import InitPredictor as initPredictorMsg

from shared_control.srv import InitPred  as initPredSrv 

from predictor_distance.srv import DistancePredictor as distancePredictorSrv
from predictor_assistance.srv import AssistancePredictor as assistancePredictorSrv
# from myo.srv import ResetMyo as ResetMyoSrv

import copy
import math
import sys
import time
import numpy as np
from enum import IntEnum
from control_manip.srv import InitObj as initobjSrv
from control_manip.msg import Goal as goalMsg
from control_manip.msg import GoalArray as goalarrayMsg
from control_manip.msg import Objects as objectsMsg
from control_manip.msg import Status as statusMsg


from shared_control.srv import InitPred  as initPredSrv
from trajectron.srv import Trajectory, GoalProb
from queue import Queue
import Clik
import Goal
# import RobotKinematics as rk
import UserInput
import Utils
import PrintFile
import PotentialFunction2 as pf

from franka_share_control.srv import cartMove, VelMove, Execute
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Path
import os
import pickle

class UserCommand(IntEnum):
    #UserCommand
    HOME = 0
    TWIST = 1
    PICK = 2
    PLACE = 3
    FINISH = 4
    EXECUTE = 5
    STOP = 6

class SharedControl:
    """
    Shared Control class \n
    Args:
        potential_params: list of potential field parameters in this order [threshold distance, repulsive gain, attractive gain]
        config_params: list of CLIK parameters in this order [delta time, vmax, diag, goal_radius]
        teleop: True if teleoperation mode is used, False otherwise
        predictor_type: "distance" if Distance predictor is used, "max_ent_ioc" if Max EntIOC is used
        robot_type: name of the robot
        grasp: True if grasp mode is used, False otherwise
        user_input_type: name of user input control type
        index_test: index of the test
        DH_params: list of Denavit-Hartenberg parameters in this order [q,a,d,alpha]
        name_user_test: name of the user
        dynamic_system: type of system, True if system is dynamic (asyncro), False otherwise
        gripper_active: True if there is a gripper, False otherwise
        escape_active: True if escape points are active, False otherwise
    """
    def __init__(self, potential_params, config_params, teleop, predictor_type, 
                 robot_type, grasp, user_input_type, index_test, name_user_test,
                 dynamic_system, gripper_active, escape_active):

        self._potential_params = potential_params
        self._config_params = config_params
        self._vmax = self._config_params[1]
        self._teleop = teleop
        self._predictor_type = predictor_type
        self._robot_type = robot_type
        self._grasp = grasp
        self._user_input_type = user_input_type
        self._index_test = index_test
        self._goal_radius = config_params[3]
        self._gripper_active = gripper_active
        self.trajectron_flag = self._config_params[4]
        
        self._end_system = False
        self._distance_type = "distance"
        self._dynamic_system = dynamic_system
        self._escape_active = escape_active
        self._LEN_TWIST_VECTOR = 6

        # self._ZT = 0.08
        self._ZT = 0.1
        self.traj_prediction=None
        self.traj_past = None

        self.JOINT_NAMES = ["panda_joint{}".format(i+1) for i in range(7)]
        self._JOINTS_ORDERED = ["panda_joint{}".format(i+1) for i in range(7)]

        #Variables for dynamic system: only obstacles can change, not goals
        if(self._dynamic_system):
            self._dynamic_first_lecture = True
            self._dynamic_goal = None
            self._dynamic_joints = None
            self._dynamic_obstacles = None
            self._dynamic_escape_points = None
            self._joints_target_positions = None
            self._dynamic_goal_msg = None
            self._dynamic_sub_objs = None


        #User Input, PrintFile and kinematics
        self._userinput = UserInput.UserInput(self._robot_type, self._user_input_type)
        print("user_input loaded")
        self._print_file = None
        if(self._teleop):
            self._print_file = PrintFile.PrintFile("teleop", self._index_test, name_user_test, self._user_input_type)
        elif self.trajectron_flag:
            self._print_file = PrintFile.PrintFile("Trajectron", self._index_test, name_user_test, self._user_input_type)
        else:
            self._print_file = PrintFile.PrintFile(self._predictor_type, self._index_test, name_user_test, self._user_input_type)
        # self._rob_kin = rk.RobotKinematics(self._DH_params[0], self._DH_params[1], self._DH_params[2], self._DH_params[3], self._robot_type, self._gripper_active)

        #Services        
        self._service_obj = None
        self._service_move = None
        self._service_grasp = None
        self._service_init_pred_node = None
        self._service_predictor = None
        self._reset_myo_service = None
        self._service_joints_move = None

        self._sub_joints = None
        self._j_q = np.zeros(6)
        self.ee_position = []
        self.ee_orientation = []
        self.eepose = Pose()


        self.path_name = 'data_intent_change2.json'
        # self.path_name = 'data.json'
        if os.path.exists(self.path_name):
            with open(self.path_name,'rb') as f:
                self.data_dict = pickle.load(f)
            # self.data_dict['frequency'] = 10
            # with open('data.json', 'wb') as fp:
            #     pickle.dump(self.data_dict, fp)
        else:
            self.data_dict = {}
            self.data_dict['frequency'] = 10
            self.data_dict['ee_log'] = []
            self.data_dict['object_list'] = []
            self.data_dict['goals'] = []
            self.data_dict['change_iter'] = []
            # with open('data.json', 'wb') as fp:
            #     pickle.dump(self.data_dict, fp)

    
    def initConnection(self):
        """
        Initialize connection to services
        """
        self._sub_joints = rospy.Subscriber("/joint_states", jointstateMsg, self.callbackJointStates)

        self._sub_eepose = rospy.Subscriber("/EE_pose", Pose, self.callbackEEPose)

        self._eepose_pub = rospy.Publisher("/EE_pose_traj", Pose, queue_size=1)

        self._sub_traj = rospy.Subscriber("/Traj_pred", PoseArray, self.callbackTrajPred)

        self._eepose_exe_pub = rospy.Publisher("/EE_pose_traj_exe", Pose, queue_size=1)
        
        self._traj_exe_srv = rospy.ServiceProxy("/TrajMove", cartMove)

        self._stop_srv = rospy.ServiceProxy('/Stop', cartMove)

        self._prob_viz_pub = rospy.Publisher("/ProbViz", MarkerArray, queue_size=1)

        self._rviz_past_sub = rospy.Subscriber("/rviz_traj_past", Path, self.callbackTrajPast, queue_size=1)


        if(self._dynamic_system):
            self._dynamic_sub_objs = rospy.Subscriber("/objects_msg", objectsMsg, self.callbackTableObjects)
        else:
            name_objects_srv =  "/objects_srv"
            rospy.wait_for_service(name_objects_srv)
            try:
                self._service_obj = rospy.ServiceProxy(name_objects_srv, initobjSrv, persistent=True)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" %e)
                rospy.on_shutdown("Error service")
                print("Shutdown")
                sys.exit()

        name_move_srv = '/VelMove'
        print("WAIT: " + str(name_move_srv))
        rospy.wait_for_service(name_move_srv)
        try:
            self._vel_move = rospy.ServiceProxy(name_move_srv, VelMove, persistent=True)
            print("velocity move load success")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" %e)
            rospy.on_shutdown("Error service")
            print("Shutdown")
            sys.exit()

        name_grasp_srv =  "/grasp_srv"
        print("WAIT: " + str(name_grasp_srv))
        rospy.wait_for_service(name_grasp_srv, 60)
        try:
            self._service_grasp = rospy.ServiceProxy(name_grasp_srv, Execute, persistent=True)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" %e)
            rospy.on_shutdown("Error service")
            print("Shutdown")
            sys.exit()
        
        if self.trajectron_flag == True:
            name_goal_prob_srv =  "/Goal_prob"
            print("WAIT: " + str(name_goal_prob_srv))
            rospy.wait_for_service(name_goal_prob_srv, 60)
            try:
                self._score_prob = rospy.ServiceProxy(name_goal_prob_srv, GoalProb, persistent=True)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" %e)
                rospy.on_shutdown("Error service")
                print("Shutdown")
                sys.exit()


    def initPredictorConnection(self):
        """
        Initialize connection to predictor service
        """
        name_init_pred_srv = "/init_prediction_srv"
        rospy.wait_for_service(name_init_pred_srv, 60)
        try:
            self._service_init_pred_node = rospy.ServiceProxy(name_init_pred_srv, initPredSrv, persistent=True)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" %e)
            rospy.on_shutdown("Error service")
            print("Shutdown")
            sys.exit()

        if(self._predictor_type == self._distance_type):
            name_predictor_srv =  "/predictor_distance_srv"
            rospy.wait_for_service(name_predictor_srv, 60)
            try:
                self._service_predictor = rospy.ServiceProxy(name_predictor_srv, distancePredictorSrv, persistent=True)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" %e)
                rospy.on_shutdown("Error service")
                print("Shutdown")
                sys.exit()
        else:
            name_predictor_srv = "/predictor_assistance_srv"
            rospy.wait_for_service(name_predictor_srv, 60)
            try:
                self._service_predictor = rospy.ServiceProxy(name_predictor_srv, assistancePredictorSrv, persistent=True)
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" %e)
                rospy.on_shutdown("Error service")
                print("Shutdown")
                sys.exit()


    def start(self):
        """
        Start system
        """
        self.initConnection()

        while(not self._end_system):
            if(self._teleop):
                print("teleop")
                self.teleopExecute()
            else:
                if(self.trajectron_flag):
                    print("trajectron system")
                    self.TrajectronExecute()                
                else:
                    print("maxent ioc system")
                    self.staticExecute()

        print("Finish: turn off the system!")
        rospy.signal_shutdown("Fine")

    
    def teleopExecute(self):
        """
        Teleoperation version of the system
        """

        ## change of intent experiment ###
        goals = []

        init_q = self._j_q
        while len(self.ee_position) == 0:
            pass
        actual_position_ee = self.ee_position
        pose_msg = self.eepose
        actual_pose_matrix_ee = Utils.PoseMsgToMatrix(pose_msg)

        response_obj = self.serviceObject(True)     
        
        print("Actual position EE: " +str(actual_position_ee))
        
        actual_position_gripper = np.zeros(3)
        if(self._gripper_active):
            actual_position_gripper = np.array(self.ee_position)
            print("Actual position gripper: " +str(actual_position_gripper))

        if(self._index_test != 0):
            self._print_file.newFile(self._index_test)
        if(self._gripper_active):
            self._print_file.write_with_title(actual_position_gripper, "Start")
        else:
            self._print_file.write_with_title(actual_position_ee, "Start")
        self._print_file.end_block()

        #Goal array
        goal_list = list()
        targets_position = list()
        goal_msg = response_obj.objects.goals
        goal_list, targets_position = Utils.getGoal(goal_msg)
        goal_pos_list = [g.getCenterPosition() for g in goal_list]
        print("Num goals: " +str(len(goal_list)))

        self._print_file.write("Goals list")
        for goal in goal_list:
            self._print_file.write_with_title(goal.getCenterPosition(), goal.getID())
        self._print_file.end_block()

        #array of distances from ee
        init_dist = []
        dist_for_change = []
        for i in range(len(targets_position)):
            if(self._gripper_active):
                dist = Utils.computeDistanceXY(actual_position_gripper, targets_position[i])
                init_dist.append(dist)
                dist_for_change.append(Utils.computeDistance(actual_position_gripper, goal_pos_list[i]))
            else:
                dist = Utils.computeDistanceXY(actual_position_gripper, targets_position[i])
                init_dist.append(dist)
                dist_for_change.append(Utils.computeDistance(actual_position_gripper, goal_pos_list[i]))

        traj_log = Queue()
        op_count = 0

        #Use InitPred srv to initialize predictor node
        #It is possible to leave EE_matrix because both predictors do not consider the z-axis
        ee_pose_msg = Utils.matrixToPoseMsg(actual_pose_matrix_ee)
        
        it = 0 #number of iterations
        att_q = init_q #for while cicle->current vector q
        distance = float('inf') #Init distance

        #init distribution: all goals are same probability
        goal_distrib = []
        for i in range(len(goal_list)):
            prob = 1./len(goal_list)
            goal_distrib.append(prob)
            print("Index: " + str(i) + " associated to: " +str(goal_list[i].getID()))
        #Index of goal with the highest probability
        index_max = 0

        #Time to reach target: timer start after first user command
        start_system = None

        z_actual = None

        change_iter = -1
        last_user_input = np.zeros(self._LEN_TWIST_VECTOR)

        print("Ready to move to goal")
        while(not ((distance < self._goal_radius) and (z_actual < self._ZT))):
            #User input twist
            twist_user_input = np.zeros(self._LEN_TWIST_VECTOR)

            twist_user_input = self._userinput.getTwist()
            while(np.linalg.norm(twist_user_input) == 0):
                if np.sqrt((last_user_input**2).sum()) < 0.01:
                    self._stop_srv.call(self.ee_position+self.ee_orientation)
                else:
                    last_user_input = last_user_input*0.1
                    self._vel_move.call(last_user_input.tolist()[:3])

                
                if(self._userinput.getCommand() != UserCommand.TWIST and (self._userinput.getCommand() != None)):
                    rospy.logerr("FAIL")
                    print(self._userinput.getCommand())
                    flag, distance, z_actual = self.returnOrFinish(targets_position, distance, z_actual)
                    self._print_file.write("TEST FAILED")
                    self._print_file.close()
                    self._index_test += 1
                    return
                
                twist_user_input = self._userinput.getTwist()
            twist_user_input = Utils.setTwist(twist_user_input, self._vmax)
            print("Twist User: " +str(twist_user_input))

            #Start timer
            if(it == 0):
                start_system = time.time()
            
            # if(not self._cart_move(dest_pose_msg.tolist())):
            #     rospy.logerr("Error movement!")
            #     rospy.on_shutdown("Error movement")
            #     sys.exit()

            if(not self._vel_move(twist_user_input.tolist()[:3])):
                rospy.logerr("Error movement!")
                rospy.on_shutdown("Error movement")
                sys.exit()
            last_user_input = twist_user_input

            actual_position_gripper = self.ee_position

            for i in range(len(init_dist)):
                if(self._gripper_active):
                    dist = Utils.computeDistanceXY(actual_position_gripper, targets_position[i])
                    dist_for_change[i] = Utils.computeDistance(actual_position_gripper, goal_pos_list[i])
                    init_dist[i] = dist
                else:
                    dist = Utils.computeDistanceXY(actual_position_gripper, targets_position[i])
                    init_dist[i] = dist
                    dist_for_change[i] = Utils.computeDistance(actual_position_gripper, goal_pos_list[i])

            index_max = np.argmin(np.array(init_dist))
            distance = init_dist[index_max]

            index_max_c = np.argmin(np.array(init_dist))
            distance_c = dist_for_change[index_max_c]
            print("distance_c:",distance_c)
            z_actual = actual_position_gripper[2]
            
            #Print on file
            self._print_file.write_with_title(twist_user_input[0:3], "user_twist")
            # if(np.linalg.norm(new_final_twist) != 0):
            #     self._print_file.write_with_title(new_final_twist[0:3], "final_twist")
            if(self._gripper_active):
                self._print_file.write_with_title(actual_position_gripper[0:3], "position_gripper")
            else:
                self._print_file.write_with_title(actual_position_ee[0:3], "position_ee")
            self._print_file.write_with_title(it, "iteration")
            self._print_file.write_with_title(goal_distrib, "distribution")
            self._print_file.end_block()

            print("Actual position: " +str(actual_position_ee))
            print("Gripper position: " +str(actual_position_gripper))
            #print("List distance: " +str(init_dist))
            print("Distance: " +str(distance))
            print("IT: " +str(it))
            it += 1
            #rospy.sleep(0.1)
        #END WHILE
        
        #Time
        end_system = time.time()
        interval = end_system - start_system
        print("Time: " + str(interval))
        print("Number of iteration: " +str(it))
        print("Distance final to goal: " +str(distance))

        self._print_file.write_with_title(interval, "time")
        self._print_file.end_block()

        #Grasp, pick and place
        goal_obj_selected = goal_list[index_max]
        print("Goal ID: " +str(goal_obj_selected.getID()))

        self._print_file.write_with_title(goal_obj_selected.getID(), "ID goal")
        self._print_file.close()
        self._index_test += 1


        #################### data recording #############################
        self.data_dict["ee_log"].append(self.traj_past)
        self.data_dict['object_list'].append(np.stack(goal_pos_list, axis=0))
        self.data_dict['change_iter'].append(change_iter)
        goals.append(index_max_c)
        self.data_dict['goals'].append(goals)
        with open(self.path_name, 'wb') as fp:
            pickle.dump(self.data_dict, fp)
        #################### data recording #############################

        self.pickPlaceRoutine(goal_obj_selected, actual_pose_matrix_ee)




    def TrajectronExecute(self):
        """
        Teleoperation version of the system
        """
        init_q = self._j_q
        while len(self.ee_position) == 0:
            pass
        ee_trajectory = []
        actual_position_ee = self.ee_position
        pose_msg = self.eepose
        actual_pose_matrix_ee = Utils.PoseMsgToMatrix(pose_msg)

        response_obj = self.serviceObject(True)     
        
        print("Actual position EE: " +str(actual_position_ee))
        
        actual_position_gripper = np.zeros(3)
        if(self._gripper_active):
            actual_position_gripper = np.array(self.ee_position)
            print("Actual position gripper: " +str(actual_position_gripper))

        if(self._index_test != 0):
            self._print_file.newFile(self._index_test)
        if(self._gripper_active):
            self._print_file.write_with_title(actual_position_gripper, "Start")
        else:
            self._print_file.write_with_title(actual_position_ee, "Start")
        self._print_file.end_block()

        #Goal array
        goal_list = list()
        targets_position = list()
        goal_msg = response_obj.objects.goals
        goal_list, targets_position = Utils.getGoal(goal_msg)
        goal_pos_list = [g.getCenterPosition() for g in goal_list]
        print("Num goals: " +str(len(goal_list)))

        self._print_file.write("Goals list")
        for goal in goal_list:
            self._print_file.write_with_title(goal.getCenterPosition(), goal.getID())
        self._print_file.end_block()

        #array of distances from ee
        init_dist = []
        for i in range(len(targets_position)):
            if(self._gripper_active):
                dist = Utils.computeDistanceXY(actual_position_gripper, targets_position[i])
                init_dist.append(dist)
            else:
                dist = Utils.computeDistanceXY(actual_position_gripper, targets_position[i])
                init_dist.append(dist)

        pot_func = pf.PotentialFunction(self._potential_params[0],self._potential_params[1],self._potential_params[2], self._potential_params[3], 
                                [], init_q, goal_pos_list, [], self._print_file, self._gripper_active, self._escape_active)


        op_count = 0
        traj_log = Queue()

        it = 0 #number of iterations
        att_q = init_q #for while cicle->current vector q
        distance = float('inf') #Init distance

        #init distribution: all goals are same probability
        goal_distrib = []
        equal_distrib = []
        for i in range(len(goal_list)):
            prob = 1/len(goal_list)*0.01
            equal_distrib.append(prob)
            goal_distrib.append(prob)
            print("Index: " +str(i) + " associated to: " +str(goal_list[i].getID()))
        #Index of goal with the highest probability
        index_max = 0
        exe_start_time = 0
        exe_end_time = 0

        #Time to reach target: timer start after first user command
        start_system = None
        last_user_input = np.zeros(self._LEN_TWIST_VECTOR)

        z_actual = None

        print("Ready to move to goal")
        while(not ((distance < self._goal_radius) and (z_actual < self._ZT))):
            self._eepose_pub.publish(self.eepose)
            #User input twist
            flag = False
            twist_user_input = np.zeros(self._LEN_TWIST_VECTOR)
            
            twist_user_input = self._userinput.getTwist()
            while(np.linalg.norm(twist_user_input) == 0):
                if np.sqrt((last_user_input**2).sum()) < 0.01:
                    self._stop_srv.call(self.ee_position+self.ee_orientation)
                else:
                    last_user_input = last_user_input*0.1
                    self._vel_move.call(last_user_input.tolist()[:3])
                                
                if(self._userinput.getCommand() != UserCommand.TWIST and (self._userinput.getCommand() != None)):
                    print(self._userinput.getCommand())
                    exe_start_time = time.time()
                    flag, distance, z_actual = self.returnOrFinish(targets_position, distance, z_actual)
                    exe_end_time = time.time()
                    if flag == True:
                        break
                    rospy.logerr("FAIL")
                    self._print_file.write("TEST FAILED")
                    self._print_file.close()
                    self._index_test += 1
                    return
                
                twist_user_input = self._userinput.getTwist()
            if flag == True:
                actual_position_gripper = self.ee_position
                actual_position_ee = self.ee_position

                for i in range(len(init_dist)):
                    if(self._gripper_active):
                        dist = Utils.computeDistanceXY(actual_position_gripper, targets_position[i])
                        init_dist[i] = dist
                    else:
                        dist = Utils.computeDistanceXY(actual_position_gripper, targets_position[i])
                        init_dist[i] = dist

                index_max = np.argmin(np.array(init_dist))
                distance = init_dist[index_max]
                z_actual = actual_position_gripper[2]
                continue

            t1 = time.time()
            twist_user_input = Utils.setTwist(twist_user_input, self._vmax)
            print("Twist User: " +str(twist_user_input))

            #Compute twist_ca from potential field
            k_g = max(np.max(np.array(goal_distrib))*1.3, 0.1)
            # weight = min([max(list(goal_distrib)),0.9])
            print("k_g:",k_g)

            twist_ca = np.zeros(self._LEN_TWIST_VECTOR)
            if sum(goal_distrib) == 0 or np.isnan(np.array(goal_distrib)).sum()>=1:
                k_g = 0.0
                print("Twist CA: " +str(twist_ca))

                a_g = 0.0
                print("a_g: " +str(a_g))
            else:

                twist_ca = pot_func.getCATwist(self.ee_position, self._vmax, goal_distrib)
                print("Twist CA: " +str(twist_ca))

                # agreement = min(np.arccos((twist_ca*twist_user_input)[0:3].sum()/(self._vmax)**2)/np.pi, 0.8)
                a_g = max((twist_ca*twist_user_input)[0:3].sum()/(self._vmax**2), 0)
                print("a_g: " +str(a_g))
            
            self._print_file.write_with_title(k_g, "k_g")
            k_g_prime = np.sqrt(k_g*a_g)
            print("k_g_prime:",k_g_prime)
            self._print_file.write_with_title(k_g_prime, "k_g_prime")

            try:
                if self.traj_prediction is not None and self.traj_past is not None:
                    tar_traj = self.traj_prediction
                    mask_z = tar_traj[:,-1] < self.ee_position[-1]
                    traj_pos = tar_traj[mask_z,:][:4]
                    traj_pos = traj_pos[-1]
                    twist_tr = Utils.setTwist(traj_pos-self.ee_position, self._vmax)
                    # twist_tr = traj_pos-self.ee_position
                    # a_tr = min(np.arccos((twist_tr*twist_user_input)[0:3].sum()/(self._vmax)**2)/np.pi, 0.8)
                    a_tr = max((twist_tr*twist_user_input)[0:3].sum()/(self._vmax**2),0)
                    # k_tr = min(max(0.2,self.traj_past.shape[0]/(self.traj_past.shape[0]+self.traj_prediction.shape[0])),0.7)
                    k_tr = min(max(0.1,self.traj_prediction.shape[0]/(self.traj_past.shape[0]+self.traj_prediction.shape[0])),0.7)
                    
 
                    # k_tr = max(0.1,self.traj_prediction.shape[0]/(self.traj_past.shape[0]+self.traj_prediction.shape[0]))
                else:
                    twist_tr = np.zeros(twist_user_input.shape)
                    a_tr = 0.0
                    k_tr = 0.0
            except:
                twist_tr = np.zeros(twist_user_input.shape)
                a_tr = 0.0
                k_tr = 0.0
            self._print_file.write_with_title(k_tr, "k_tr")
            k_tr_prime = np.sqrt(a_tr*k_tr)
            self._print_file.write_with_title(k_tr_prime, "k_tr_prime")
            #Final twist
            final_twist = Utils.setTwist(twist_ca*k_g_prime + twist_user_input*(1-k_g_prime)+(1-k_g_prime)*k_tr_prime*twist_tr, self._vmax)

            # final_twist = Utils.setTwist(twist_ca + twist_user_input, self._vmax)      
            print("Twist Final: " +str(final_twist))
            print("k_tr:",k_tr)

            # dest_pose_msg[0:3] += twist_user_input[0:3] * 0.1
            # dest_pose_msg[0:3] += final_twist[0:3] * 0.1

            t2 = time.time()

            #Start timer
            if(it == 0):
                start_system = time.time()
            
            # if(not self._cart_move(dest_pose_msg.tolist())):
            #     rospy.logerr("Error movement!")
            #     rospy.on_shutdown("Error movement")
            #     sys.exit()
            if(not self._vel_move(final_twist.tolist()[:3])):
                rospy.logerr("Error movement!")
                rospy.on_shutdown("Error movement")
                sys.exit()
            last_user_input = final_twist
            
            t3 = time.time()
            current_pose = self.eepose
            # if op_count % 10 == 0:
            ##################### viz_trajectory ##################################
            # traj_log.put(current_pose)
            # if len(traj_log.queue) > 10:
            #     traj_log.get()
            # if op_count >= 10 and op_count % 10 == 0:
            #     req = PoseArray()
            #     req.poses = list(traj_log.queue)
            #     response = self.trajectron_srv.call(req)
            ##################### viz_trajectory ##################################
            try:
                if op_count >= 4:
                    posearray = self.nparray2PoseArray(np.stack(goal_pos_list,axis=0))
                    prob = self._score_prob.call(posearray)
                    goal_distrib = prob.probs
                    # goal_distrib = np.array(prob.probs)
                    # goal_distrib = (goal_distrib/goal_distrib.sum()).tolist()
                    print(prob)
                    prob_viz =  MarkerArray()
                    for i in range(len(goal_distrib)):
                        marker = Marker()
                        marker.header.frame_id = 'world'
                        marker.type = Marker.TEXT_VIEW_FACING
                        marker.ns = "object"
                        marker.pose.orientation.w = 1.0
                        marker.id = goal_list[i].getID()
                        marker.scale.x = 0.05
                        marker.scale.y = 0.05
                        marker.scale.z = 0.05
                        marker.color.b = 25
                        marker.color.g = 0
                        marker.color.r = 25
                        marker.color.a = 1
                        pos = posearray.poses[i]
                        pos.position.x += 0.05
                        pos.position.y += 0.05
                        pos.position.z += 0.05
                        marker.text ="{:.2f}".format(goal_distrib[i])
                        marker.pose = pos
                        prob_viz.markers.append(marker)
                    self._prob_viz_pub.publish(prob_viz)
            except:
                pass

            op_count += 1

            t4 = time.time()


            # if(self._gripper_active):
            actual_position_gripper = self.ee_position
            actual_position_ee = self.ee_position

            for i in range(len(init_dist)):
                if(self._gripper_active):
                    dist = Utils.computeDistanceXY(actual_position_gripper, targets_position[i])
                    init_dist[i] = dist
                else:
                    dist = Utils.computeDistanceXY(actual_position_gripper, targets_position[i])
                    init_dist[i] = dist

            index_max = np.argmin(np.array(init_dist))
            distance = init_dist[index_max]
            z_actual = actual_position_gripper[2]
            
            #Print on file
            self._print_file.write_with_title(twist_user_input[0:3], "user_twist")
            # if(np.linalg.norm(new_final_twist) != 0):
            #     self._print_file.write_with_title(new_final_twist[0:3], "final_twist")
            if(self._gripper_active):
                self._print_file.write_with_title(actual_position_gripper[0:3], "position_gripper")
            else:
                self._print_file.write_with_title(actual_position_ee[0:3], "position_ee")
            self._print_file.write_with_title(it, "iteration")
            self._print_file.write_with_title(goal_distrib, "distribution")
            self._print_file.end_block()
            # print("velo cal time:",t2-t1)
            # print("move time:",t3-t2)
            # print("marker dist print:",t4-t3)

            print("Actual position: " +str(actual_position_ee))
            print("Gripper position: " +str(actual_position_gripper))
            #print("List distance: " +str(init_dist))
            print("Distance: " +str(distance))
            print("IT: " +str(it))
            it += 1
            #rospy.sleep(0.1)
        #END WHILE
        
        #Time
        end_system = time.time()
        interval = end_system - start_system
        print("Time: " + str(interval))
        # print("EXE Time: " + str(exe_end_time-exe_start_time))
        print("Number of iteration: " +str(it))
        print("Distance final to goal: " +str(distance))

        self._print_file.write_with_title(interval, "time")
        self._print_file.write_with_title(str(exe_end_time-exe_start_time), "EXE time")
        self._print_file.end_block()

        #Grasp, pick and place
        goal_obj_selected = goal_list[index_max]
        print("Goal ID: " +str(goal_obj_selected.getID()))

        self._print_file.write_with_title(goal_obj_selected.getID(), "ID goal")
        self._print_file.close()
        self._index_test += 1

        # if(self._grasp):
        self.pickPlaceRoutine(goal_obj_selected, actual_pose_matrix_ee)
        prob_viz=MarkerArray()
        for i in range(len(goal_distrib)):
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.type = Marker.TEXT_VIEW_FACING
            marker.ns = "object"
            marker.pose.orientation.w = 1.0
            marker.id = goal_list[i].getID()
            prob_viz.markers.append(marker)
        self._prob_viz_pub.publish(prob_viz)
        # self.data_dict["ee_log"].append(self.traj_past)
        # with open(self.path_name, 'wb') as fp:
        #     pickle.dump(self.data_dict, fp)


    def staticExecute(self):
        """
        Static version of the execution of the routine for picking up the object with collision avoidance and predictor nodes
        """

        #Get init joints values, goals, obstacles and escape points
        init_q = self._j_q
        while len(self.ee_position) == 0:
            pass
        actual_position_ee = self.ee_position
        pose_msg = self.eepose
        actual_pose_matrix_ee = Utils.PoseMsgToMatrix(pose_msg)

        response_obj = self.serviceObject(True)     
        
        print("Actual position EE: " +str(actual_position_ee))
        
        actual_position_gripper = np.zeros(3)
        if(self._gripper_active):
            actual_position_gripper = np.array(self.ee_position)
            print("Actual position gripper: " +str(actual_position_gripper))

        if(self._index_test != 0):
            self._print_file.newFile(self._index_test)
        if(self._gripper_active):
            self._print_file.write_with_title(actual_position_gripper, "Start")
        else:
            self._print_file.write_with_title(actual_position_ee, "Start")
        self._print_file.end_block()

        #Fixed Obstacles array
        obs_position = Utils.getListPoints(response_obj.objects.obstacles)

        #Escape points
        escape_points = Utils.getListPoints(response_obj.objects.escape_points)
        
        #Goal array
        goal_list = list()
        targets_position = list()
        goal_msg = response_obj.objects.goals
        goal_list, targets_position = Utils.getGoal(goal_msg)
        goal_pos_list = [g.getCenterPosition() for g in goal_list]
        print("Num goals: " +str(len(goal_list)))

        self._print_file.write("Goals list")
        for goal in goal_list:
            self._print_file.write_with_title(goal.getCenterPosition(), goal.getID())
        self._print_file.end_block()

        #array of distances from ee
        init_dist = []
        for i in range(len(targets_position)):
            if(self._gripper_active):
                dist = Utils.computeDistanceXY(actual_position_gripper, targets_position[i])
                init_dist.append(dist)
            else:
                dist = Utils.computeDistanceXY(actual_position_gripper, targets_position[i])
                init_dist.append(dist)
                        
        #Init predictor service connection
        self.initPredictorConnection()

        #Use InitPred srv to initialize predictor node
        #It is possible to leave EE_matrix because both predictors do not consider the z-axis
        ee_pose_msg = Utils.matrixToPoseMsg(actual_pose_matrix_ee)
        if(not self.serviceInitPred(goal_msg, ee_pose_msg)):
            rospy.logerr("Error init predictor node")
            rospy.on_shutdown("Error")
            return

        #Init other params
        pot_func = pf.PotentialFunction(self._potential_params[0],self._potential_params[1],self._potential_params[2], self._potential_params[3], 
                                obs_position, init_q, goal_pos_list, escape_points, self._print_file, self._gripper_active, self._escape_active)

        it = 0 #number of iterations
        att_q = init_q #for while cicle->current vector q
        distance = float('inf')
        
        #init distribution: all goals are same probability
        goal_distrib = []
        for i in range(len(goal_list)):
            prob = 1./len(goal_list)
            goal_distrib.append(prob)
            print("Index: " +str(i) + " associated to: " +str(goal_list[i].getID()))
        #Index of goal with the highest probability
        index_max = 0

        #Time to reach target: timer start after first user command
        start_system = None
        last_user_input = np.zeros(self._LEN_TWIST_VECTOR)
        z_actual = None

        print("Ready to move to goal")
        while(not ((distance < self._goal_radius) and (z_actual < self._ZT))):
            #Final twist for ee
            final_twist = np.zeros(self._LEN_TWIST_VECTOR)
            #Twist from target prediction and assistance node
            twist_a = np.zeros(self._LEN_TWIST_VECTOR)
            #Twist from collision avoidance
            twist_ca = np.zeros(self._LEN_TWIST_VECTOR)
            #User input twist
            twist_user_input = np.zeros(self._LEN_TWIST_VECTOR)

            twist_user_input = self._userinput.getTwist()
            #Always wait user twist command
            while(np.linalg.norm(twist_user_input) == 0):
                if np.sqrt((last_user_input**2).sum()) < 0.01:
                    self._stop_srv.call(self.ee_position+self.ee_orientation)
                else:
                    last_user_input = last_user_input*0.1
                    self._vel_move.call(last_user_input.tolist()[:3])
                                
                if(self._userinput.getCommand() != UserCommand.TWIST and (self._userinput.getCommand() != None)):
                    rospy.logerr("FAIL")
                    print(self._userinput.getCommand())
                    self.returnOrFinish(targets_position, distance, z_actual)
                    self._print_file.write("TEST FAILED")
                    self._print_file.close()
                    self._index_test += 1
                    return
                
                twist_user_input = self._userinput.getTwist()
            twist_user_input = Utils.setTwist(twist_user_input, self._vmax)
            print("Twist User: " +str(twist_user_input))
            # t1 = time.time()
            #Start timer
            if(it == 0):
                start_system = time.time()

            #Prediction and assistance service to compute probability and assistance twist
            if(self._predictor_type != self._distance_type):
                #There is a user command
                if(np.linalg.norm(twist_user_input) != 0):
                    #Service to target assistance
                    response_assist = self.serviceAssistancePredictor(twist_user_input*1.3, twist_ca, actual_pose_matrix_ee)
                    goal_distrib = response_assist.distr_prob
                    index_max = response_assist.index_max
                    #twist_a = Utils.twistMsgToArray(response_assist.assisted_twist)
                    #velocity saturation
                    #twist_a = Utils.setTwist(twist_a, self._vmax) 
                    #print("Twist A: " +str(twist_a))
            print("Distribution: " +str(goal_distrib))

            #Compute twist_ca from potential field
            twist_ca = pot_func.getCATwist(self.ee_position, self._vmax, goal_distrib)
            print("Twist CA: " +str(twist_ca))

            #Final twist
            final_twist = Utils.setTwist(twist_ca + twist_user_input, self._vmax)      
            print("Twist Final: " +str(final_twist))

            # pose_msg = self.eepose
            # dest_pose_msg = [pose_msg.position.x,pose_msg.position.y,pose_msg.position.z,pose_msg.orientation.x,pose_msg.orientation.y,pose_msg.orientation.z,pose_msg.orientation.w]
            # dest_pose_msg = np.array(dest_pose_msg)

            # dest_pose_msg[0:3] += final_twist[0:3] * 0.1
            # t2 = time.time()


            #Start timer
            if(it == 0):
                start_system = time.time()
            
            # t1 = time.time()
            
            if(not self._vel_move(final_twist.tolist()[:3])):
                rospy.logerr("Error movement!")
                rospy.on_shutdown("Error movement")
                sys.exit()
            # print("exe_time:",time.time()-t1)
            
            # if(not self._cart_move(dest_pose_msg.tolist())):
            #     rospy.logerr("Error movement!")
            #     rospy.on_shutdown("Error movement")
            #     sys.exit()
            # t3 = time.time()
            last_user_input = final_twist

            actual_position_gripper = self.ee_position

            for i in range(len(init_dist)):
                if(self._gripper_active):
                    dist = Utils.computeDistanceXY(actual_position_gripper, targets_position[i])
                    init_dist[i] = dist
                else:
                    dist = Utils.computeDistanceXY(actual_position_gripper, targets_position[i])
                    init_dist[i] = dist

            index_max = np.argmin(np.array(init_dist))
            distance = init_dist[index_max]
            z_actual = actual_position_gripper[2]

            #Print on file
            self._print_file.write_with_title(twist_user_input[0:3], "user_twist")
            self._print_file.write_with_title(twist_ca[0:3], "avoidance_twist")
            #self._print_file.write_with_title(twist_a[0:3], "assistance_twist")
            # if(np.linalg.norm(new_final_twist) == 0):
            #     self._print_file.write_with_title(final_twist[0:3], "final_twist")
            # else:
            #     self._print_file.write_with_title(new_final_twist[0:3], "final_twist")
            if(self._gripper_active):
                self._print_file.write_with_title(actual_position_gripper[0:3], "position_gripper")
            else:
                self._print_file.write_with_title(actual_position_ee[0:3], "position_ee")
            self._print_file.write_with_title(it, "iteration")
            self._print_file.write_with_title(goal_distrib, "distribution")
            self._print_file.end_block()
            # print("velo cal time:",t2-t1)
            # print("move time:",t3-t2)

            print("Actual position: " +str(actual_position_ee))
            print("Gripper position: " +str(actual_position_gripper))
            #print("List distance: " +str(init_dist))
            print("Distance: " +str(distance))
            print("IT: " +str(it))
            it += 1

            #rospy.sleep(0.1)
        #END WHILE

        #Time
        end_system = time.time()
        interval = end_system - start_system
        print("Time: " + str(interval))
        print("Number of iteration: " +str(it))
        print("Distance final to goal: " +str(distance))

        self._print_file.write_with_title(interval, "time")
        self._print_file.end_block()
        
        #Grasp, pick and place
        goal_obj_selected = goal_list[index_max]
        print("Goal ID: " +str(goal_obj_selected.getID()))

        self._print_file.write_with_title(goal_obj_selected.getID(), "ID goal")
        self._print_file.close()
        self._index_test += 1

        self.pickPlaceRoutine(goal_obj_selected, actual_pose_matrix_ee)
        # self.data_dict["ee_log"].append(self.traj_past)
        # with open('data.json', 'wb') as fp:
        #     pickle.dump(self.data_dict, fp)

    
    def follow_trajectory(self, targets_position):
        traj_prediction = self.traj_prediction
        for i in range(traj_prediction.shape[0]):
            pose_msg = self.eepose
            dest_pose_msg = [pose_msg.position.x,pose_msg.position.y,pose_msg.position.z,pose_msg.orientation.x,pose_msg.orientation.y,pose_msg.orientation.z,pose_msg.orientation.w]
            dest_pose_msg[0] = traj_prediction[i,0]
            dest_pose_msg[1] = traj_prediction[i,1]
            dest_pose_msg[2] = traj_prediction[i,2]
            self._cart_move(dest_pose_msg)
            self._eepose_exe_pub.publish(self.eepose)
            actual_position_gripper = self.ee_position
            z_actual = actual_position_gripper[2]
            init_dist = []
            for i in range(len(targets_position)):
                if(self._gripper_active):
                    dist = Utils.computeDistanceXY(actual_position_gripper, targets_position[i])
                    init_dist.append(dist)
                else:
                    dist = Utils.computeDistanceXY(actual_position_gripper, targets_position[i])
                    init_dist.append(dist)

            index_max = np.argmin(np.array(init_dist))
            distance = init_dist[index_max]
            if ((distance < self._goal_radius) and (z_actual < self._ZT)):
                return distance, z_actual
        return distance, z_actual
    
    def follow_trajectory2(self, targets_position):
        stride = 4
        traj_prediction = self.traj_prediction
        i = stride
        arrived = False
        while i < traj_prediction.shape[0] - 1 or arrived == False:
            pose_msg = self.eepose
            dest_pose_msg = [pose_msg.position.x,pose_msg.position.y,pose_msg.position.z,pose_msg.orientation.x,pose_msg.orientation.y,pose_msg.orientation.z,pose_msg.orientation.w]
            # dest_pose_msg[0] = traj_prediction[4,0]
            # dest_pose_msg[1] = traj_prediction[4,1]
            # dest_pose_msg[2] = traj_prediction[4,2]
            dest_pose_msg[0] = traj_prediction[i,0]
            dest_pose_msg[1] = traj_prediction[i,1]
            dest_pose_msg[2] = traj_prediction[i,2]
            self._traj_exe_srv.call(dest_pose_msg)
            # time.sleep(0.01)
            if i < traj_prediction.shape[0] - stride:
                i += stride
            elif i < traj_prediction.shape[0] - 1:
                i += 1
            self._eepose_exe_pub.publish(self.eepose)                
            actual_position_gripper = self.ee_position
            z_actual = actual_position_gripper[2]
            dists = []
            for j in range(len(targets_position)):
                if(self._gripper_active):
                    dist = Utils.computeDistanceXY(actual_position_gripper, targets_position[j])
                    dists.append(dist)
                else:
                    dist = Utils.computeDistanceXY(actual_position_gripper, targets_position[j])
                    dists.append(dist)

            index_max = np.argmin(np.array(dists))
            distance = dists[index_max]
            if z_actual < self._ZT:
                self._stop_srv.call(self.ee_position+self.ee_orientation)
                # self._traj_exe_srv.call(self.ee_position+self.ee_orientation)
                return distance, z_actual
            if np.abs(np.array(self.ee_position) - traj_prediction[-1,:]).sum() < 1e-2:
                arrived = True
                self._stop_srv.call(self.ee_position+self.ee_orientation)
                # self._traj_exe_srv.call(self.ee_position+self.ee_orientation)
            if self._userinput.getCommand() == UserCommand.STOP:
                self._stop_srv.call(self.ee_position+self.ee_orientation)
                # self._traj_exe_srv.call(self.ee_position+self.ee_orientation)
                return distance, z_actual
        return distance, z_actual
    
    def returnOrFinish(self, targets_position,distance, z_actual):
        flag = False
        if self._userinput.getCommand() == UserCommand.EXECUTE:
            distance, z_actual = self.follow_trajectory2(targets_position)
            flag = True
        return flag, distance, z_actual


    def nparray2PoseArray(self, traj_array):
        traj_array = traj_array.reshape(-1,3)
        traj_posearray = PoseArray()
        for i in range(traj_array.shape[0]):
            term = Pose()
            term.position.x = traj_array[i,0]
            term.position.y = traj_array[i,1]
            term.position.z = traj_array[i,2]
            traj_posearray.poses.append(term)
        return traj_posearray

    
    def pickPlaceRoutine(self, goal_obj, pose_matrix_ee):
        """
        Pick and Place routine \n
        Args:
            goal_obj: Goal object
            pose_matrix_ee: ee pose matrix
        """
        '''
        while(self._userinput.getCommand() != UserCommand.PICK):
            rospy.logwarn("No PICK command: please insert PICK command!")
            rospy.sleep(1.0)
        '''

        response_grasp = self.serviceGrasp(goal_obj, pose_matrix_ee)

        # self.returnOrFinish()
        

    def serviceResetMyo(self, reset=True):
        """
        Service to reset Myo pose \n
        Args:
            reset
        Return: response
        """
        if(self._user_input_type == "myo"):
            response = self._reset_myo_service.call(reset)
            return response
        return True


    def serviceInitPred(self, goals_vector_msg, ee_pose=Pose()):
        """
        Service to initialize predictor node \n
        Args:
            goals_vector_msg: Goal msg vector
            ee_pose: ee Pose msg
        Return: response
        """
        response = self._service_init_pred_node.call(goals_vector_msg, ee_pose)
        return response
        

    def serviceObject(self, status):
        """
        Service to received goals, obstacles, fixed escape points and initial value of joints
        Args:
            status: status
        Return: response
        """
        status_msg = statusMsg()
        status_msg.ready = status
        response = self._service_obj.call(status_msg)
        print("Received info")
        return response 


    def serviceAssistancePredictor(self, user_twist, potential_twist, ee_matrix):
        """
        Service to target assistance node
        Args:
            user_twist: user twist
            potential_twist: collision avoidance twist
            ee_matrix: pose ee
        Return: response
        """
        twist_u = Utils.arrayToTwistMsg(user_twist)
        twist_pot = Utils.arrayToTwistMsg(potential_twist)
        matrix_ee = Utils.matrixToPoseMsg(ee_matrix)
        response = self._service_predictor.call(twist_u, twist_pot, matrix_ee)
        return response 


    def serviceDistancePredictor(self, distances):
        """
        Service to distance predictor node \n
        Args:
            distances: distances from ee to goals
        Return: response
        """
        response = self._service_predictor.call(distances)
        return response


    def serviceGrasp(self, goal, ee_pose):
        """
        Service to compute rank of grasping points \n
        Args:
            goal: Goal object
            ee_pose: ee_pose matrix
        Return: grasping points rank
        """
        goal_msg = goalMsg()
        goal_msg.id = goal.getID()
        goal_msg.center = Utils.matrixToPoseMsg(goal.getCenterMatrix())
        goal_msg.grasping_points = goal.getGraspingPoints()

        pose_ee_msg = Utils.matrixToPoseMsg(ee_pose)
        response = self._service_grasp.call(goal.getID())
        return response



    def callbackTableObjects(self, data):
        """
        Callback to manage objects on the table
        """
        if(self._dynamic_first_lecture):
            self._dynamic_joints = Utils.getInitJoints(data.joints)
            self._dynamic_goal, self._joints_target_positions = Utils.getGoal(data.goals)
            self._dynamic_goal_msg = data.goals
            self._dynamic_first_lecture = False
        
        self._dynamic_obstacles = Utils.getListPoints(data.obstacles)
        self._dynamic_escape_points = Utils.getListPoints(data.escape_points)


    def orderJoints(self, joints):
        keys_list = self.JOINT_NAMES
        values_list = joints.tolist()
        zip_iterator = zip(keys_list, values_list)
        dict_j = dict(zip_iterator)
        final_joints = list()
        for k in self._JOINTS_ORDERED:
            final_joints.append(dict_j[k])

        return final_joints

    
    def disorderJoints(self, joints):
        keys_list = self._JOINTS_ORDERED
        values_list = joints
        zip_iterator = zip(keys_list, values_list)
        dict_j = dict(zip_iterator)
        final_joints = list()
        for k in self.JOINT_NAMES:
            final_joints.append(dict_j[k])

        return final_joints

    def poseArray2nparray(self, pose_array_msg):
        # num_steps = should be 10
        num_steps = len(pose_array_msg.poses)
        data = []
        for i in range(num_steps):
            data.append(np.array([pose_array_msg.poses[i].position.x, pose_array_msg.poses[i].position.y, pose_array_msg.poses[i].position.z]))
        data = np.stack(data,axis=0)
        return data
    
    def path2nparray(self, pose_array_msg):
        # num_steps = should be 10
        num_steps = len(pose_array_msg.poses)
        data = []
        for i in range(num_steps):
            data.append(np.array([pose_array_msg.poses[i].pose.position.x, pose_array_msg.poses[i].pose.position.y, pose_array_msg.poses[i].pose.position.z]))
        data = np.stack(data,axis=0)
        return data

    
    def callbackJointStates(self, msg):
        self._j_q = self.disorderJoints(msg.position)
    
    def callbackTrajPred(self, msg):
        try:
            self.traj_prediction = self.poseArray2nparray(msg)
        except:
            # print("traj_log error")
            pass
    
    def callbackTrajPast(self, msg):
        try:
            self.traj_past = self.path2nparray(msg)
        except:
            pass
        
    
    def callbackEEPose(self, msg):
        self.eepose = msg
        self.ee_position = [msg.position.x, msg.position.y, msg.position.z]
        self.ee_orientation = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        
if __name__ == "__main__":
    rospy.init_node("shared_control_node_py", anonymous=True)

    mode = 2
    user_name = "test"
    if mode == 1:
        teleop_ = True
        use_trajectron = False
    elif mode == 2:
        teleop_ = False
        use_trajectron = True
    else:
        teleop_ = False
        use_trajectron = False
    
    #Get all configuration parameters
    #Potential field params
    threshold_distance = rospy.get_param("threshold_distance", 0.1)
    attractive_gain = rospy.get_param("attractive_gain", 0.05)
    repulsive_gain = rospy.get_param("repulsive_gain", 1)
    escape_gain = rospy.get_param("escape_gain", 1000)
    potential_params = [threshold_distance, repulsive_gain, attractive_gain, escape_gain]
    print("Threshold distance: " +str(potential_params[0]))
    
    #CLIK params
    delta_time = rospy.get_param("delta_time", 0.02)
    vmax = rospy.get_param("max_abs_velocity", 0.1)
    # vmax = rospy.get_param("max_abs_velocity", 1) # cart_move 
    diag = rospy.get_param("diagonal", 50)
    # goal_radius = rospy.get_param("goal_radius", 0.12)
    goal_radius = rospy.get_param("goal_radius", 0.07)

    trajectron_flag = rospy.get_param("trajectron_flag", use_trajectron)
    config_params = [delta_time, vmax, diag, goal_radius, trajectron_flag]
    print("Goal radius: " +str(goal_radius))

    #Only teleoperation
    teleop = rospy.get_param("~teleop", teleop_)
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
    name_user_test = rospy.get_param("~name_user_test", user_name)
    print("Name user: " +str(name_user_test))

    dynamic_system = rospy.get_param("~dynamic", False)
    print("Dynamic system: " +str(dynamic_system))

    gripper_active = rospy.get_param("~gripper", True)
    print("Gripper enable: " +str(gripper_active))

    escape_active = rospy.get_param("~escape", False)
    print("Escape active: " +str(escape_active))

    #Start
    shared_control = SharedControl(potential_params, config_params, teleop, predictor_type, robot_type, grasp, user_input_type, index_test,
                                                        name_user_test, dynamic_system, gripper_active, escape_active)


    shared_control.start()


    print("Finish: turn off system")
    