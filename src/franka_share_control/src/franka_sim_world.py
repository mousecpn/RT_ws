import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
import os
import glob
import random
from scipy.spatial.transform import Rotation
import roboticstoolbox as rtb

import math
from sensor_msgs.msg import JointState as jointstateMsg
from utils import setCameraOnRobotWrist, angle_axis, calculate_velocity, rand, velocity_based_control, pseudo_etasl
import rospy

from msg_builder.joint_msg_builder import joint_state_builder
from threading import Thread, Lock
from franka_share_control.srv import cartMove,VelMove,Execute
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from control_manip.srv import InitObj as initobjSrv
from control_manip.srv import Move as moveSrv
from control_manip.msg import Goal as goalMsg
from control_manip.msg import GoalArray as goalarrayMsg
from control_manip.msg import Objects as objectsMsg
from franka_share_control.msg import KeyCommand
from geometry_msgs.msg import Twist, TransformStamped, Transform
# import pygame
from queue import Queue
from trajectron.srv import Trajectory, VeloMerge
import tf2_ros
from nav_msgs.msg import Path

pandaNumDofs = 7
maxV = 0.2

ll = [-7]*pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs
#restposes for null space
init_joint_pose1=[0.0, -math.pi/4, 0.0, -3*math.pi/4, 0.0, math.pi/2, math.pi/4, 0.0, 0.0]

rp = init_joint_pose1
total_data_num = 20000

class PandaSim(object):
    def __init__(self, offset):
        self.offset = np.array(offset)
        self.LINK_EE_OFFSET = 0.05
        self.initial_offset = 0.05
        self.workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
        self._numObjects = 5
        self._urdfRoot = pd.getDataPath()
        self._blockRandom = 0.3
        # self._sdfRoot = "/home/pinhao/Desktop/franka_share_control/models/objects"
        self._sdfRoot = "/home/pinhao/Desktop/franka_sim_ws/src/franka_share_control/models/exp1"

        self.joint_pub = rospy.Publisher('/Joint_states', jointstateMsg, queue_size=1)
        self.eepose_pub = rospy.Publisher('/EE_pose', Pose, queue_size=1)
        self.cart_move_srv = rospy.Service('/CartMove', cartMove, self.handle_move_command)
        self.vel_move_srv = rospy.Service('/VelMove', VelMove, self.handle_move_vel_command)
        self.object_srv = rospy.Service('/objects_srv', initobjSrv, self.handle_objects_srv)
        self.grasp_srv = rospy.Service('/grasp_srv', Execute, self.pickPlaceRoutine)
        self._rviz_past_pub = rospy.Publisher("/rviz_traj_past", Path, queue_size=1)
        self._trajectory_follower = rospy.Service("/TrajMove", cartMove, self.handle_traj_move_command)
        self.stop_srv = rospy.Service('/Stop', cartMove, self.handle_stop_command)
        
        self.traj_pred_sub = rospy.Subscriber('/Traj_pred', PoseArray, self.traj_pred_handler)

        self.past_trajectory = Path()
        self.future_trajectory = None

        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        orn=[0.0, 0.0, 0, 1.0]
        self.init_joint_pose=[0.0, -math.pi/4, 0.0, -3*math.pi/4, 0.0, math.pi/2, math.pi/4, 0.0, 0.0]
        self.place_joint_pose=[-math.pi/2, -math.pi/4, 0.0, -3*math.pi/4, 0.0, math.pi/2, math.pi/4, 0.0, 0.0]


        p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])

        p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.620000,0.000000,0.000000,0.0,1.0)


        self.panda = p.loadURDF("franka_panda/panda.urdf", np.array([0,0,0])+self.offset, orn, useFixedBase=True, flags=flags)
        self.goal_ids, self.obstacle_ids, self.escape_ids = self.setting_objects(globalScaling=0.7)
        self.goal_ids = set(self.goal_ids)
        self.obstacle_ids = set(self.obstacle_ids)
        self.escape_ids = set(self.escape_ids)
        self.control_dt = 0.01
        self.place_poses = [-0.00018899307178799063, -0.3069845139980316, 0.48534566164016724]
        self.z_T = 0.1
        self.reset()

        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.ee_pose = Pose()

        return
    
    def traj_pred_handler(self, traj_pred):
        if len(traj_pred.poses) != 0:
            self.future_trajectory = poseArray2nparray(traj_pred)
        return

    def rviz_object_publisher(self):
        while True:
            try:
                for idx in self.goal_ids:
                    translation,orientation = p.getBasePositionAndOrientation(idx)
                    static_transformStamped = TransformStamped()

                    static_transformStamped.header.stamp = rospy.Time.now()
                    static_transformStamped.header.frame_id = "world"
                    static_transformStamped.child_frame_id = "object{}".format(idx)

                    static_transformStamped.transform.translation.x = translation[0]
                    static_transformStamped.transform.translation.y = translation[1]
                    static_transformStamped.transform.translation.z = translation[2]
                    static_transformStamped.transform.rotation.x = orientation[0]
                    static_transformStamped.transform.rotation.y = orientation[1]
                    static_transformStamped.transform.rotation.z = orientation[2]
                    static_transformStamped.transform.rotation.w = orientation[3]

                    self.broadcaster.sendTransform(static_transformStamped)
                    time.sleep(0.001)
            except:
                pass
        return

    def handle_objects_srv(self, flag):
        objects = objectsMsg()
        goals = goalarrayMsg()
        for idx in self.goal_ids:
            translation,orientation = p.getBasePositionAndOrientation(idx)
            pose_msg = goalMsg()
            pose_msg.id = idx
            pose_msg.center.position.x = translation[0]
            pose_msg.center.position.y = translation[1]
            pose_msg.center.position.z = translation[2]
            pose_msg.center.orientation.x = orientation[0]
            pose_msg.center.orientation.y = orientation[1]
            pose_msg.center.orientation.z = orientation[2]
            pose_msg.center.orientation.w = orientation[3]
            grasp_point = PoseStamped()
            grasp_point.pose.position.x = translation[0]
            grasp_point.pose.position.y = translation[1]
            grasp_point.pose.position.z = translation[2] + self.z_T
            grasp_point.pose.orientation.x = 0.515299
            grasp_point.pose.orientation.y = 0.478671
            grasp_point.pose.orientation.z = -0.50345
            grasp_point.pose.orientation.w = 0.501875
            pose_msg.grasping_points.append(grasp_point)
            goals.goal.append(pose_msg)
        objects.goals = goals

        return objects
    
    def handle_stop_command(self, req):
        for i in range(pandaNumDofs):
            p.setJointMotorControl2(self.panda, i, p.VELOCITY_CONTROL, targetVelocity=0.0, force=5 * 240.)
        return True
    
    def handle_move_command(self, req):
        ee_pose = req.ee_pose
        if len(ee_pose) == 3:
            return self.keyboard_move(ee_pose)
        else:
            return self.keyboard_move(ee_pose[:3], ee_pose[3:])
    
    def handle_move_vel_command(self, req):
        ee_vel = req.ee_vel
        if len(ee_vel) == 3:
            return self.move_vel(ee_vel)
        else:
            return self.move_vel(ee_vel[:3], ee_vel[3:])
    
    def handle_traj_move_command(self, req):
        ee_pose = req.ee_pose
        if len(ee_pose) == 3:
            return self.traj_follower(ee_pose)
        else:
            return self.traj_follower(ee_pose[:3], ee_pose[3:])
    
    def JointStatePub(self):
        while True:
            names = []
            joint_states = []
            for j in range(p.getNumJoints(self.panda)):
                names.append(str(p.getJointInfo(self.panda,j)[1], encoding ="utf-8"))
                joint_states.append(p.getJointState(self.panda,j)[0])
            joint_states_msg = joint_state_builder(names, joint_states)
            self.joint_pub.publish(joint_states_msg)
            time.sleep(0.01)
        return

    def EEPosePub(self):
        while True:
            translation = p.getLinkState(self.panda,11)[4]
            orientation = p.getLinkState(self.panda,11)[5]
            pose_msg = Pose()
            pose_msg.position.x = translation[0]
            pose_msg.position.y = translation[1]
            pose_msg.position.z = translation[2]
            pose_msg.orientation.x = orientation[0]
            pose_msg.orientation.y = orientation[1]
            pose_msg.orientation.z = orientation[2]
            pose_msg.orientation.w = orientation[3]
            self.eepose_pub.publish(pose_msg)
            self.ee_pose = pose_msg
            time.sleep(0.01)
        return
    
    def rviz_past_publisher(self):
        while True:
            self.past_trajectory.header.stamp = rospy.Time.now()
            self.past_trajectory.header.frame_id = 'world'
            self._rviz_past_pub.publish(self.past_trajectory)
            time.sleep(0.01)

    def setting_objects(self,globalScaling):
        goal_ids = []
        obstacle_ids = []
        escape_ids = []
        # blue cube
        files = os.listdir(self._sdfRoot)
        for file in files:
            uid = p.loadSDF(os.path.join(self._sdfRoot,file),globalScaling=globalScaling)
            goal_ids.append(uid[0])
        return goal_ids, obstacle_ids, escape_ids
    

    
    def random_approaching_orn(self, tar_id):
        obj_pose, obj_orn = p.getBasePositionAndOrientation(tar_id)
        ee_pose, ee_orn = p.getLinkState(self.panda,11)[4:6]
        R_Mat = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3,3)

        z_direct = (np.array(obj_pose)-np.array(ee_pose))
        z_direct += (2*np.random.rand(3)-1)*0.1
        z_direct = z_direct/np.linalg.norm(z_direct)
        w = np.random.rand()
        z_direct = w * z_direct + (1-w) * np.array([0,0,-1])

        aux_axis = R_Mat[:,0]
        y_direct = np.cross(z_direct, aux_axis)
        y_direct += (2*np.random.rand(3)-1)*0.01
        y_direct = y_direct/np.linalg.norm(y_direct)

        x_direct = np.cross(y_direct, z_direct)
        x_direct = x_direct/np.linalg.norm(x_direct)

        matrix = np.array(
            [x_direct, y_direct, z_direct]).T
        
        grasp_orn = Rotation.from_matrix(matrix).as_quat()
        return matrix, grasp_orn
    
    def reset(self):
        index = 0
        for i in range(pandaNumDofs):
            p.setJointMotorControl2(self.panda, i, p.VELOCITY_CONTROL, targetVelocity=0, force=5 * 240.)
        for j in range(p.getNumJoints(self.panda)):
            p.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.panda, j)
            init_joint_pose = np.array(self.init_joint_pose)#+\
                #np.array([rand(), rand()*0.5, rand()*0.3, rand()*0.3, rand(), rand()*0.2, 0, 0 , 0])

            jointName = info[1]
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC):
                p.resetJointState(self.panda, j, init_joint_pose[index]) 
                index=index+1
            if (jointType == p.JOINT_REVOLUTE):
                p.resetJointState(self.panda, j, init_joint_pose[index]) 
                index=index+1
        self.init_pose, self.init_orn = p.getLinkState(self.panda,11)[4:6]
        # if reset_obejects==True:
        #     urdfList = self.get_random_object(self._numObjects, False)
        #     self.objectUids = set(self.randomly_place_objects(urdfList))
        self.panda_control = rtb.models.Panda()
        self.gripper_homing()
    
    def place_pose(self):
        # while True:
        jointPoses = np.array(self.place_joint_pose)
        success = True
        for i in range(pandaNumDofs):
            p.setJointMotorControl2(self.panda, i, p.POSITION_CONTROL, jointPoses[i],force=5 * 240., maxVelocity=0.6)
        time.sleep(5)
        print(p.getLinkState(self.panda,11)[4])
        # if np.abs(np.array(p.getLinkState(self.panda,11)[4]) - np.array(self.place_joint_pose)).sum() < 1e-3:
        #     break
        return True
    
    def gripper_homing(self):
        p.setJointMotorControl2(self.panda, 9, p.POSITION_CONTROL, 0.04, force=5 * 240., maxVelocity=maxV)
        p.setJointMotorControl2(self.panda, 10, p.POSITION_CONTROL, 0.04, force=5 * 240., maxVelocity=maxV)
        # print(p.getJointState(self.panda,0)[1])
        if abs(p.getJointState(self.panda, 9)[0] - 0.04) < 1e-5:
            return True
        time.sleep(1)
        return False
    
    def move(self, pos, orn=None):
        pos = np.array(pos)
        # orn = p.getQuaternionFromEuler([0.,0,0])  #math.pi/2.
        if orn is None:
            orn=[1.0, 0.0, 0.0, 0.0]

        while True:
            joint_pose = []
            joint_vel = []
            for i in range(7):
                pos_i, vel_i, _, _ = p.getJointState(self.panda,i)
                joint_pose.append(pos_i)
                joint_vel.append(vel_i)
            target_vel,arrived = calculate_velocity(p, self.panda_control, np.array(joint_pose), pos, orn)
            success = True
            try:
                for i in range(pandaNumDofs):
                    p.setJointMotorControl2(self.panda, i, p.VELOCITY_CONTROL, targetVelocity=target_vel[i], force=5 * 240.) #+target_vel[i]*rand()
                if np.abs(np.array(p.getLinkState(self.panda,11)[4]) - np.array(pos)).mean() < 1e-2:
                    break
            except:
                return False
        for i in range(pandaNumDofs):
            p.setJointMotorControl2(self.panda, i, p.VELOCITY_CONTROL, targetVelocity=0.0, force=5 * 240.)
        return success
    
    def keyboard_move(self, pos, orn=None):
        pos = np.array(pos)
        if orn is None:
            orn=[1.0, 0.0, 0.0, 0.0]
            # pos[-1] += self.LINK_EE_OFFSET
        # else:
        #     R_Mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
            # pos -= self.LINK_EE_OFFSET * R_Mat[:,2]
        joint_pose = []
        joint_vel = []
        for i in range(7):
            pos_i, vel_i, _, _ = p.getJointState(self.panda,i)
            joint_pose.append(pos_i)
            joint_vel.append(vel_i)

        target_vel,arrived = calculate_velocity(p, self.panda_control, np.array(joint_pose), pos, orn)
        success = True
        while True:
            try:
                for i in range(pandaNumDofs):
                    p.setJointMotorControl2(self.panda, i, p.VELOCITY_CONTROL, targetVelocity=target_vel[i], force=5 * 240.) #+target_vel[i]*rand()
                time.sleep(0.1)
                for i in range(pandaNumDofs):
                    p.setJointMotorControl2(self.panda, i, p.VELOCITY_CONTROL, targetVelocity=0.0, force=5 * 240.)
                # if np.abs(np.array(p.getLinkState(self.panda,11)[4]) - np.array(pos)).sum() < 1e-3:
                break
            except:
                return False
        # visualization
        current_pos = PoseStamped()
        current_pos.header.stamp = rospy.Time.now()
        current_pos.header.frame_id = 'world'
        current_pos.pose = self.ee_pose
        self.past_trajectory.poses.append(current_pos)
        return success
    

    def traj_follower(self, pos, orn=None):
        pos = np.array(pos)
        if orn is None:
            orn=[1.0, 0.0, 0.0, 0.0]
            # pos[-1] += self.LINK_EE_OFFSET
        # else:
        #     R_Mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
            # pos -= self.LINK_EE_OFFSET * R_Mat[:,2]
        joint_pose = []
        joint_vel = []
        for i in range(7):
            pos_i, vel_i, _, _ = p.getJointState(self.panda,i)
            joint_pose.append(pos_i)
            joint_vel.append(vel_i)

        target_vel, arrived = calculate_velocity(p, self.panda_control, np.array(joint_pose), pos, orn)
        if arrived:
            for i in range(pandaNumDofs):
                p.setJointMotorControl2(self.panda, i, p.VELOCITY_CONTROL, targetVelocity=0.0, force=5 * 240.)
        else:
            for i in range(pandaNumDofs):
                p.setJointMotorControl2(self.panda, i, p.VELOCITY_CONTROL, targetVelocity=target_vel[i], force=5 * 240.)
        
        # visualization
        current_pos = PoseStamped()
        current_pos.header.stamp = rospy.Time.now()
        current_pos.header.frame_id = 'world'
        current_pos.pose = self.ee_pose
        self.past_trajectory.poses.append(current_pos)
        return arrived
    
    def move_vel(self, pos_vel, ang_vel=[0.0,0.0,0.0]):
        # xyz -> zyx
        list(ang_vel).reverse()
        joint_pose = []
        joint_vel = []
        for i in range(7):
            pos_i, vel_i, _, _ = p.getJointState(self.panda,i)
            joint_pose.append(pos_i)
            joint_vel.append(vel_i)
        
        target_vel = velocity_based_control(p, self.panda_control, np.array(joint_pose), np.array(pos_vel), np.array(ang_vel))
        success = True
        try:
            for i in range(pandaNumDofs):
                p.setJointMotorControl2(self.panda, i, p.VELOCITY_CONTROL, targetVelocity=target_vel[i], force=5 * 240.) #+target_vel[i]*rand()
            # time.sleep(self.control_dt)
            time.sleep(0.1)
            # visualization
            current_pos = PoseStamped()
            current_pos.header.stamp = rospy.Time.now()
            current_pos.header.frame_id = 'world'
            current_pos.pose = self.ee_pose
            self.past_trajectory.poses.append(current_pos)
            # for i in range(pandaNumDofs):
            #     p.setJointMotorControl2(self.panda, i, p.VELOCITY_CONTROL, targetVelocity=0.0, force=5 * 240.)
        except:
            print("no success")
            success = False
        return success
    


    def approaching(self, tar_id, orn=None):
        if orn is None:
            orn = np.array([1.0, 0.0, 0.0, 0.0])
        pos,_ = p.getBasePositionAndOrientation(tar_id)
        pos = np.array(pos)
        pos[-1] += self.LINK_EE_OFFSET
        # orn = p.getQuaternionFromEuler([0.,0,0])  #math.pi/2.

        # approching
        while True:
            jointPoses = p.calculateInverseKinematics(self.panda,11, pos, orn, ll, ul, jr, rp, maxNumIterations=10)
            for i in range(pandaNumDofs):
                p.setJointMotorControl2(self.panda, i, p.POSITION_CONTROL, jointPoses[i], force=5 * 240., maxVelocity=0.2)
            if np.abs(np.array(p.getLinkState(self.panda,11)[4]) - np.array(pos)).sum() < 1e-3:
                # print(np.abs(np.array(p.getLinkState(self.panda,11)[4]) - np.array(pos)).sum())
                break
        
        pos[-1] -= self.LINK_EE_OFFSET-0.02
        # approching
        
        while True:
            jointPoses = p.calculateInverseKinematics(self.panda,11, pos, orn, ll, ul,
                jr, rp, maxNumIterations=10)
            for i in range(pandaNumDofs):
                p.setJointMotorControl2(self.panda, i, p.POSITION_CONTROL, jointPoses[i], force=5 * 240., maxVelocity=0.2)
                # print(p.getJointState(self.panda,i)[1])
            if np.abs(np.array(p.getLinkState(self.panda,11)[4]) - np.array(pos)).sum() < 1e-3:
                # print(np.abs(np.array(p.getLinkState(self.panda,11)[4]) - np.array(pos)).sum())
                break
        return True
    

    def ready_pose2(self):
        while True:
            joint_pose = []
            joint_vel = []
            ready_pose = self.init_pose#+(2*np.random.rand(3)-1)*0.02
            for i in range(7):
                pos_i, vel_i, _, _ = p.getJointState(self.panda,i)
                joint_pose.append(pos_i)
                joint_vel.append(vel_i)
            target_vel, arrived = calculate_velocity(p, self.panda_control, np.array(joint_pose), np.array(ready_pose), np.array(self.init_orn))
            success = True
            for i in range(pandaNumDofs):
                p.setJointMotorControl2(self.panda, i, p.VELOCITY_CONTROL, targetVelocity=target_vel[i],force=5 * 240., maxVelocity=0.1)
            if np.abs(np.array(p.getLinkState(self.panda,11)[4]) - np.array(self.init_pose)).sum() > 1e-2:
                success = success & False
            else:
                for i in range(pandaNumDofs):
                    p.setJointMotorControl2(self.panda, i, p.VELOCITY_CONTROL, targetVelocity=0.0,force=5 * 240.)
                break
        self.past_trajectory = Path()
        return success


    def grasp(self, tar_id):
        p.setJointMotorControl2(self.panda, 9, p.POSITION_CONTROL, 0.0, force=5 * 240., maxVelocity=maxV)
        p.setJointMotorControl2(self.panda, 10, p.POSITION_CONTROL, 0.0, force=5 * 240., maxVelocity=maxV)
        if bool(p.getContactPoints(bodyA=self.panda,bodyB=tar_id)):
            # p.setJointMotorControl2(self.panda, 9, p.VELOCITY_CONTROL, targetVelocity=0.0, force=5 * 240., maxVelocity=maxV)
            # p.setJointMotorControl2(self.panda, 10, p.VELOCITY_CONTROL, targetVelocity=0.0, force=5 * 240., maxVelocity=maxV)
            return True
        return False

    # def object_setup():

    #     return
    def remove(self,tar_id):
        p.removeBody(tar_id)
        self.goal_ids.remove(tar_id)
        return
    
    def ee_pose_sub(self,ee_pose_msg):
        self.ee_pose = ee_pose_msg

    def trajectron_visualizer(self, req):
        p.removeAllUserDebugItems()
        response = self.trajectron_srv.call(req)
        prediction = poseArray2nparray(response.prediction)
        visualize_trajectory(prediction)
        return response
    
    def pickPlaceRoutine(self, req):
        tarid = req.tarid
        self.approaching(tarid)
        self.grasp(tarid)
        # self.ready_pose2()
        self.move(self.place_poses)
        self.gripper_homing()
        self.ready_pose2()
        self.remove(tarid)
        if len(self.goal_ids) == 0:
            self.goal_ids, self.obstacle_ids, self.escape_ids = self.setting_objects(globalScaling=0.7)
            self.goal_ids = set(self.goal_ids)
            self.obstacle_ids = set(self.obstacle_ids)
            self.escape_ids = set(self.escape_ids)
        # self.past_trajectory = Path()
        return True



def visualize(pos,orn):
    R_Mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
    x_axis = R_Mat[:,0]
    x_end_p = (np.array(pos) + np.array(x_axis*5)).tolist()
    x_line_id = p.addUserDebugLine(pos,x_end_p,[1,0,0])# y 轴
    y_axis = R_Mat[:,1]
    y_end_p = (np.array(pos) + np.array(y_axis*5)).tolist()
    y_line_id = p.addUserDebugLine(pos,y_end_p,[0,1,0])# z轴
    z_axis = R_Mat[:,2]
    z_end_p = (np.array(pos) + np.array(z_axis*5)).tolist()
    z_line_id = p.addUserDebugLine(pos,z_end_p,[0,0,1])

def stepSimulation(iter):
    for k in range(iter):
        p.stepSimulation()
    return

class Watchdog():
    def __init__(self,limit=200):
        self.count = 0
        self.limit=limit
        return
    
    def error(self):
        self.count+=1
        if self.count>self.limit:
            return True
        return False
    
    def reset(self):
        self.count=0
        return

def keyboard_detection(panda, velo=0.2):
    pub_command = rospy.Publisher("/user_command", KeyCommand, queue_size=1)
    twist_msg = Twist()
    command_msg = KeyCommand()
    while True:
        command_msg.header.stamp = rospy.Time.now()

        #Create zero twist message
        twist_msg.linear.x = 0
        twist_msg.linear.y = 0
        twist_msg.linear.z = 0
        twist_msg.angular.x = 0
        twist_msg.angular.y = 0
        twist_msg.angular.z = 0
        g = p.getKeyboardEvents()

        if ord('x') in g:
            command_msg.command = 5
        elif 32 in g:
            command_msg.command = 6
        else:
            command_msg.command = command_msg.TWIST
        # if len(g.keys()) == 0:
        #     continue
        if p.B3G_UP_ARROW in g:
            twist_msg.linear.x = velo
        
        if p.B3G_LEFT_ARROW in g:
            twist_msg.linear.y = velo
        
        if p.B3G_DOWN_ARROW in g:
            twist_msg.linear.x = -velo
        
        if p.B3G_RIGHT_ARROW in g:
            twist_msg.linear.y = -velo
        
        if ord('a') in g:
            twist_msg.linear.z = velo
        
        if ord('z') in g:
            twist_msg.linear.z = -velo

        if ord('h') in g:
            panda.ready_pose2()

        
        command_msg.twist = twist_msg
        pub_command.publish(command_msg)
        time.sleep(0.02)

    
def poseArray2nparray(pose_array_msg):
    # num_steps = should be 10
    num_steps = len(pose_array_msg.poses)
    data = []
    for i in range(num_steps):
        data.append(np.array([pose_array_msg.poses[i].position.x, pose_array_msg.poses[i].position.y, pose_array_msg.poses[i].position.z]))
    data = np.stack(data,axis=0)
    return data

def visualize_trajectory(prediction):
    threads = []
    tt1 = time.time()
    for i in range(prediction.shape[0]-1):
        threads.append(Thread(target=p.addUserDebugLine,args=(prediction[i].tolist(),prediction[i+1].tolist(),[1,0,0])))
        # p.addUserDebugLine(prediction[i].tolist(),prediction[i+1].tolist(),[1,0,0])
    tt2 = time.time()
    for i in range(prediction.shape[0]-1):
        threads[i].start()
    tt3 = time.time()
    return


if __name__=="__main__":
    rospy.init_node("franka_sim")
    p.connect(p.GUI)
    # p.connect(p.DIRECT)
    # p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)
    
    p.setAdditionalSearchPath(pd.getDataPath())

    p.resetDebugVisualizerCamera(cameraDistance=1,cameraYaw=0,\
                                cameraPitch=-40,cameraTargetPosition=[-0.5,-0.9,1.5])
    timeStep=1./100.
    steps = 5
    p.setTimeStep(timeStep)
    p.setRealTimeSimulation(1)

    p.setGravity(0,0,-9.8)

    traj_log = Queue()

    rospy.Rate(2)
    
    panda = PandaSim([0,0,0])
    t0 = Thread(target=keyboard_detection,name='keyboard_detection', args=(panda,))
    t0.start()


    t1 = Thread(target=panda.JointStatePub, name='joint_state_pubilisher')
    t1.start()
    # t1.join()

    t2 = Thread(target=panda.EEPosePub,name='ee_pose_pubilisher')
    t2.start()

    t3 = Thread(target=panda.rviz_object_publisher, name='object_rviz')
    t3.start()

    t4 = Thread(target=panda.rviz_past_publisher, name='rviz_past_pubilisher')
    t4.start()


    pose_sub = rospy.Subscriber('/EE_pose', Pose, panda.ee_pose_sub)

    service = rospy.ServiceProxy('/VelMove', VelMove)
    service2 = rospy.ServiceProxy('/trajectron', Trajectory)
    service3 = rospy.ServiceProxy('/Velo_merge', VeloMerge) 

    eepose_pub = rospy.Publisher("/EE_pose_traj", Pose, queue_size=1)



        
    # rospy.spin()
    # panda.pickPlaceRoutine(8)
    while True:
        stepSimulation(1)
        time.sleep(0.01)
    # t1.join()
