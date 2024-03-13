import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt
from model.trajectron import Trajectron
import rospy
from trajectron.srv import Trajectory, GoalProb, VeloMerge
from geometry_msgs.msg import PoseArray,Pose, PoseStamped
from queue import Queue
from threading import Thread
import time
import matplotlib.pyplot as plt
from nav_msgs.msg import Path,OccupancyGrid

def derivative_of(x, dt=1):
    not_nan_mask = ~np.isnan(x)
    masked_x = x[not_nan_mask]

    if masked_x.shape[-1] < 2:
        return np.zeros_like(x)

    dx = np.full_like(x, np.nan)
    dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=(masked_x[1] - masked_x[0])) / dt
    # dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=0.0) / dt

    return dx

def derivatives_of(x, dt=1, radian=False):
    timestep, dim = x.shape
    dxs = []
    for d in range(dim):
        dxs.append(derivative_of(x[:,d],dt))
    dxs = np.stack(dxs,axis=-1)
    return dxs

class trajectron_service:
    def __init__(self, trajectron, ph):
        self.trajectron = trajectron
        self.ph = ph
        self.dt = 0.1
        self.ph_limit = 100
        self.scale = 15
        self.dist = None
        self.z_T = 0.04
        self.traj_log = Queue()
        self._eepose_sub = rospy.Subscriber("/EE_pose_traj", Pose, self.trajectory_prediction_asyn,queue_size=1)
        # self._eepose_sub_for_dist = rospy.Subscriber("/EE_pose_traj", Pose, self.dist_prediction_asyn)
        self._eepose_exe_sub = rospy.Subscriber("/EE_pose_traj_exe", Pose, self.exe_log, queue_size=1)
        self._traj_pub = rospy.Publisher("/Traj_pred", PoseArray, queue_size=1)
        self.op_count = 0
        self.traj_msg = PoseArray()
        # self._rviz_path_pub = rospy.Publisher("/rviz_traj_pred", Path)
        self._rviz_prediction_pub = rospy.Publisher("/rviz_traj_pred", Path, queue_size=1)
        self._rviz_map_pub = rospy.Publisher("/rviz_heatmap", OccupancyGrid, queue_size=1)

        self.x_range = [0, 1.0]
        self.y_range = [-0.5, 0.5]
        self.user_std = 0.2

        # work space
        x_min = self.x_range[0]
        x_max = self.x_range[1]
        y_min = self.y_range[0]
        y_max = self.y_range[1]
        self.resolution = 0.08
        self.analysis_resolution = 0.01
        self.XYZ_ana = torch.meshgrid([torch.arange(x_min, x_max, self.analysis_resolution), torch.arange(y_min, y_max, self.analysis_resolution),torch.FloatTensor([self.z_T])],indexing='ij')
        self.XYZ = torch.meshgrid([torch.arange(x_min, x_max, self.resolution), torch.arange(y_min, y_max, self.resolution),torch.FloatTensor([self.z_T])],indexing='ij')
        self.prob_grid = None
        self.grid_msg = OccupancyGrid()
        self.grid_msg.info.resolution = self.resolution
        self.grid_msg.info.width = torch.arange(x_min, x_max, self.resolution).shape[0]
        self.grid_msg.info.height = torch.arange(y_min, y_max, self.resolution).shape[0]
        self.grid_msg.info.origin.position.x = 0.0
        self.grid_msg.info.origin.position.y = -0.5
        self.grid_msg.info.origin.position.z = 0.0
        self.grid_msg.info.origin.orientation.x = 0.0
        self.grid_msg.info.origin.orientation.y = 0.0
        self.grid_msg.info.origin.orientation.z = 0.0
        self.grid_msg.info.origin.orientation.w = 1.0
        self.grid_msg.data = [0 for i in range(self.grid_msg.info.width*self.grid_msg.info.height)]
        return

    def traj_publisher(self):
        while True:
            self._traj_pub.publish(self.traj_msg)
            time.sleep(0.01)
    
    def rviz_traj_publisher(self):
        while True:
            traj_msg = Path()
            traj_msg.header.stamp = rospy.Time.now()
            traj_msg.header.frame_id = 'world'
            traj_msg.poses = self.PoseArray2PoseStampedlist(self.traj_msg)
            self._rviz_prediction_pub.publish(traj_msg)
            time.sleep(0.01)
    
    def rviz_map_publisher(self):
        while True:
            self.grid_msg.header.stamp = rospy.Time.now()
            self.grid_msg.header.frame_id = 'map'
            self._rviz_map_pub.publish(self.grid_msg)
            time.sleep(0.01) 
    
        
    def PoseArray2PoseStampedlist(self, posearray):
        posestampeds = []
        for i in range(len(posearray.poses)):
            term = PoseStamped()
            term.header.frame_id = 'world'
            term.pose = posearray.poses[i]
            posestampeds.append(term)
        return posestampeds

    def exe_log(self, pos):
        self.traj_log.put(pos)
        if len(self.traj_log.queue) > 10:
            self.traj_log.get()

    def trajectory_prediction_asyn(self, pos):
        if len(self.traj_log.queue) > 0:
            current_pos = np.array([pos.position.x, pos.position.y, pos.position.z])
            last_pos = np.array([self.traj_log.queue[-1].position.x, self.traj_log.queue[-1].position.y, self.traj_log.queue[-1].position.z])
            if np.sqrt(((current_pos-last_pos)**2).sum()) > 0.2:
                self.traj_log = Queue()
        self.traj_log.put(pos)
        self.op_count += 1
        if len(self.traj_log.queue) > 10:
            self.traj_log.get()
        # validate whether arrive the destination
        if pos.position.z < self.z_T:
            return 
        if len(self.traj_log.queue) >= 4 and self.op_count % 2 == 0:
            pose_array_msg = PoseArray()
            pose_array_msg.poses = list(self.traj_log.queue)
            trajectory = self.poseArray2Traj(pose_array_msg)
            with torch.no_grad():
                # y_dist, predictions = self.trajectron.predict(trajectory,
                #                             self.ph,
                #                             num_samples=1,
                #                             z_mode=True,
                #                             gmm_mode=True,
                #                             all_z_sep=False,
                #                             full_dist=False,
                #                             dist=True)
                y_dist, v_dist, predictions = self.trajectron.predict2(trajectory,
                                self.z_T*self.scale,
                                num_samples=1,
                                z_mode=True,
                                gmm_mode=True,
                                all_z_sep=False,
                                full_dist=False,
                                dist=True,
                                ph_limit=100)
            self.dist = y_dist
            self.v_dist = v_dist

            # if self.prediction_draw is not None:
            #     self.prediction_draw.remove()
            self.traj_msg = self.nparray2PoseArray(predictions/self.scale)
            means = self.dist.mus
            n_component, _, timesteps,_, dim = means.shape

            n = 0
            t = timesteps - 1
            # for t in range(timesteps):
            nt_gmm = self.dist.get_for_node_at_time(n, t)

            search_grid = torch.stack(self.XYZ_ana, dim=2).view(-1, 3).float().to(means.device)*self.scale
            score = torch.exp(nt_gmm.log_prob(search_grid))+1e-6 #/torch.exp(nt_gmm.log_prob(nt_gmm.mus.reshape(-1,3))).max()
            if torch.isnan(score).sum() >= 1:
                print("error")

            self.prob_grid = torch.mean(score, dim=0)
            # print("latent function:",self.prob_grid.sum())
            self.prob_grid = torch.nn.functional.adaptive_avg_pool2d(self.prob_grid.reshape(1,1,int((self.x_range[1]-self.x_range[0])/self.analysis_resolution),  int((self.y_range[1]-self.y_range[0])/self.analysis_resolution)), (self.grid_msg.info.height,self.grid_msg.info.width))
            self.prob_grid = self.prob_grid/self.prob_grid.sum()
            self.prob_grid = self.prob_grid.reshape(-1)
            # try:
            for i in range(self.grid_msg.info.height):
                for j in range(self.grid_msg.info.width):
                    self.grid_msg.data[i*self.grid_msg.info.width+j] = -int(self.prob_grid[j*self.grid_msg.info.width+i]*255)+127
            # except:
            #     pass
        return
    
    def dist_prediction_asyn(self, pos):
        if self.op_count >= 3 and self.op_count%1 == 0:
            pose_array_msg = PoseArray()
            pose_array_msg.poses = list(self.traj_log.queue)
            trajectory = self.poseArray2Traj(pose_array_msg)
            with torch.no_grad():
                y_dist, _, _ = self.trajectron.predict2(trajectory,
                                self.z_T*self.scale,
                                num_samples=20,
                                z_mode=False,
                                gmm_mode=False,
                                all_z_sep=False,
                                full_dist=True,
                                dist=True,
                                ph_limit=100)
            self.dist = y_dist
            # if self.prediction_draw is not None:
            #     self.prediction_draw.remove()
            means = self.dist.mus
            n_component, _, timesteps,_, dim = means.shape

            n = 0
            t = timesteps - 1
            # for t in range(timesteps):
            nt_gmm = self.dist.get_for_node_at_time(n, t)

            search_grid = torch.stack(self.XYZ, dim=2).view(-1, 3).float().to(means.device)*self.scale
            score = torch.exp(nt_gmm.log_prob(search_grid))#/torch.exp(nt_gmm.log_prob(nt_gmm.mus.reshape(-1,3))).max()

            self.prob_grid = torch.mean(score, dim=0).reshape(score[0].shape)
            self.prob_grid = self.prob_grid/self.prob_grid.sum()
        return

    
    def trajectory_prediction(self, req):
        pose_array_msg = req.past
        trajectory = self.poseArray2Traj(pose_array_msg)
        with torch.no_grad():
            # y_dist, predictions = self.trajectron.predict(trajectory,
            #                             self.ph,
            #                             num_samples=1,
            #                             z_mode=True,
            #                             gmm_mode=True,
            #                             all_z_sep=False,
            #                             full_dist=False,
            #                             dist=True)
            y_dist, v_dist, predictions = self.trajectron.predict2(trajectory,
                            self.z_T*self.scale,
                            num_samples=1,
                            z_mode=True,
                            gmm_mode=True,
                            all_z_sep=False,
                            full_dist=False,
                            dist=True,
                            ph_limit=100)
        self.dist = y_dist
        predictions_msg = self.nparray2PoseArray(predictions/self.scale)
        return predictions_msg

    
    def poseArray2Traj(self, pose_array_msg):
        # num_steps = should be 10
        num_steps = len(pose_array_msg.poses)
        data = []
        for i in range(num_steps):
            data.append(np.array([pose_array_msg.poses[i].position.x, pose_array_msg.poses[i].position.y, pose_array_msg.poses[i].position.z]))
        term = np.stack(data, axis=0)*self.scale
        vel_term = derivatives_of(term, dt=self.dt)
        acc_term = derivatives_of(vel_term, dt=self.dt)
        data = np.concatenate((term,vel_term,acc_term),axis=-1)
        first_history_index = torch.LongTensor(np.array([0])).cuda()
        x = data[2:,:]
        y = np.zeros((12,9))
        std = np.array([3,3,3,2,2,2,1,1,1])
        # std = np.array([1,1,1,1,1,1,1,1,1])

        rel_state = np.zeros_like(x[0])
        rel_state[0:3] = np.array(x)[-1, 0:3]

        x_st = np.where(np.isnan(x), np.array(np.nan), (x - rel_state) / std)
        y_st = np.where(np.isnan(y), np.array(np.nan), y / std)
        x_t = torch.tensor(x, dtype=torch.float).unsqueeze(0).cuda()
        y_t = torch.tensor(y, dtype=torch.float).unsqueeze(0).cuda()
        x_st_t = torch.tensor(x_st, dtype=torch.float).unsqueeze(0).cuda()
        y_st_t = torch.tensor(y_st, dtype=torch.float).unsqueeze(0).cuda()
        batch = (first_history_index, x_t, y_t[...,3:6], x_st_t, y_st_t[...,3:6])
        return batch

    def nparray2PoseArray(self, traj_array):
        traj_array = traj_array.reshape(-1,3)
        traj_posearray = PoseArray()
        ph = traj_array.shape[0]
        for i in range(ph):
            term = Pose()
            term.position.x = traj_array[i,0]
            term.position.y = traj_array[i,1]
            term.position.z = traj_array[i,2]
            traj_posearray.poses.append(term)
        return traj_posearray
    


    def velo_merge(self, req):
        v_user = req.velo
        v_dist = self.v_dist

        n = 0
        nt_gmm = v_dist.get_at_time(0)
        gmm_2d = nt_gmm.splating()
        pred_var = nt_gmm.get_covariance_matrix().squeeze().cpu().numpy()
        user_var = self.user_std**2 * np.eye(3)

        K = pred_var @ np.linalg.inv(pred_var + user_var)

        v_pred = nt_gmm.mode().squeeze().cpu().numpy()

        v_merge = v_pred.reshape(3,1) + K @ (v_user - v_pred).reshape(3,1)

        return [v_merge.astype('float32').squeeze().tolist()]

        
    
    def score(self, req):
        # goal_position: PoseArray
        goals_msg = req.GoalArray
        pred_dist = self.dist
        goal_position = self.poseArray2nparray(goals_msg)
        
        means = pred_dist.mus
        n_component, _, timesteps,_, dim = means.shape
        goal_position = torch.FloatTensor(goal_position*self.scale).float().to(means.device)
        num_goals = goal_position.shape[0]

        # n = 0
        # mode_t_list = []
        # t = timesteps - 1
        # # for t in range(timesteps):
        # nt_gmm = pred_dist.get_for_node_at_time(n, t)
        prob = []

        search_grid = torch.stack(self.XYZ, dim=2).view(-1, 3).float().to(means.device)*self.scale
        # score = torch.exp(nt_gmm.log_prob(search_grid))#/torch.exp(nt_gmm.log_prob(nt_gmm.mus.reshape(-1,3))).max()

        # prob_grid = torch.mean(score, dim=0).reshape(score[0].shape)
        # prob_grid = prob_grid/prob_grid.sum()

        x_mask = (search_grid[:,None, 0]-goal_position[None,:, 0]).abs() <= self.scale*0.08
        y_mask = (search_grid[:,None, 1]-goal_position[None,:, 1]).abs() <= self.scale*0.08

        search_grid.unsqueeze(1).repeat(1,num_goals,1)[x_mask*y_mask, :]
        for i in range(num_goals):
            dist = ((search_grid[x_mask[:,i]*y_mask[:,i], :] - goal_position[i:i+1,:])**2).sum(-1).sqrt()
            dist = 1.0/(dist+1e-6)
            weight = dist/dist.sum()
            prob.append(np.float32((self.prob_grid.reshape(-1)[x_mask[:,i]*y_mask[:,i]]*weight).sum().cpu()))


        return [prob]
    
    def poseArray2nparray(self, pose_array_msg):
        # num_steps = should be 10
        num_steps = len(pose_array_msg.poses)
        data = []
        for i in range(num_steps):
            data.append(np.array([pose_array_msg.poses[i].position.x, pose_array_msg.poses[i].position.y, pose_array_msg.poses[i].position.z]))
        data = np.stack(data,axis=0)
        return data


if __name__=='__main__':
    rospy.init_node("trajectron")
    from argument_parser import args

    if not torch.cuda.is_available() or args.device == 'cpu':
        args.device = torch.device('cpu')
    else:
        if torch.cuda.device_count() == 1:
            # If you have CUDA_VISIBLE_DEVICES set, which you should,
            # then this will prevent leftover flag arguments from
            # messing with the device allocation.
            args.device = 'cuda:0'

        args.device = torch.device(args.device)

    if args.eval_device is None:
        args.eval_device = torch.device('cpu')

    # This is needed for memory pinning using a DataLoader (otherwise memory is pinned to cuda:0 by default)
    torch.cuda.set_device(args.device)


    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval
    hyperparams['offline_scene_graph'] = args.offline_scene_graph
    hyperparams['node_freq_mult_train'] = args.node_freq_mult_train
    hyperparams['node_freq_mult_eval'] = args.node_freq_mult_eval
    hyperparams['scene_freq_mult_train'] = args.scene_freq_mult_train
    hyperparams['scene_freq_mult_eval'] = args.scene_freq_mult_eval
    hyperparams['scene_freq_mult_viz'] = args.scene_freq_mult_viz
    hyperparams['augment'] = args.augment
    hyperparams['override_attention_radius'] = args.override_attention_radius
    best_ade = 1000
    
    hyperparams["frequency"] = 10


    log_writer = None
    model_dir = None

    trajectron = Trajectron(hyperparams,
                            log_writer,
                            args.device)
    model = torch.load(args.model_path)

    trajectron.model.node_modules = model
    trajectron.set_annealing_params()

    max_hl = hyperparams['maximum_history_length']
    ph = hyperparams['prediction_horizon']
    trajectron.model.to(args.device)
    trajectron.model.eval()

    traj_service = trajectron_service(trajectron, ph=50)

    t1 = Thread(target=traj_service.traj_publisher, name='traj_pubilisher')
    t1.start()

    t2 = Thread(target=traj_service.rviz_traj_publisher, name='rviz_prediction_pubilisher')
    t2.start()

    # t3 = Thread(target=traj_service.rviz_past_publisher, name='rviz_past_pubilisher')
    # t3.start()

    t3 = Thread(target=traj_service.rviz_map_publisher, name='rviz_map_pubilisher')
    t3.start()

    traj_pred = rospy.Service('/trajectron', Trajectory, traj_service.trajectory_prediction)
    traj_pred = rospy.Service('/Goal_prob', GoalProb, traj_service.score)
    traj_pred = rospy.Service('/Velo_merge', VeloMerge, traj_service.velo_merge)
    print("trajectron loaded successfully")

    

    rospy.spin()

