import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
import pickle


root = "/home/pinhao/Desktop/stat"
modes = ["teleop", "max_ent_ioc","Trajectron"]
pos_dict = {"teleop":[],
            "max_ent_ioc":[],
            "Trajectron":[]}
colors = {"teleop":"blue",
            "max_ent_ioc":"red",
            "Trajectron":"green"}
color = ['red', 'green', 'blue', 'darkorange']

goals ={0: [0.6032694342304135,0.037405109183549494],
        1: [0.5871035359003376,-0.12813010091286664],
        2: [0.4109234640533609,-0.13171441891091618],
        3: [0.4118878533343156,0.035611931062185444]
        }

def analysis_trajectory(cubes_idx = 3):
    user_dirs = os.listdir(root)
    for user in user_dirs:
        if user=="test":
            continue
        for m in modes:
            data_dir = os.path.join(root,user,m,'keyboard')
            if not os.path.exists(data_dir):
                continue
            data_files = os.listdir(data_dir)
            
            for data_file in data_files:
                if data_file.split("_")[-2] != str(cubes_idx):
                    continue
                traj = []
                file_path = os.path.join(data_dir,data_file)
                f = open(file_path,"r")
                for line in f:
                    # print(line)
                    if line.split(":")[0]=="position_gripper":
                        pos = line.split(":")[-1][1:-1].split(" ")
                        pos = list(map(float,pos))
                        pos = np.array(pos)
                        traj.append(pos)
                if np.sqrt(((pos[:2] - np.array(goals[cubes_idx]))**2).sum())>0.2:
                    continue
                if len(traj)>20:
                    pos_dict[m].append(np.stack(traj,axis=0))

    fig = plt.figure()
    ax = plt.subplot(projection='3d')
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-0.3, 0.1)
    ax.set_zlim(0, 0.8)

    for key in pos_dict.keys():
        for traj in pos_dict[key]:
            vis_data = traj
            ax.plot3D(vis_data[:,0], vis_data[ :,1], vis_data[:,2], c=colors[key])
            # ax.scatter3D(vis_data[::2,0], vis_data[::2,1], vis_data[::2,2], s=5, c='green')
    

    x,y,z = np.indices([11,11,11])
    x = x/12.
    y = y/12. - 0.5
    z = z/12.

    cubes = (abs(goals[cubes_idx][0]-x)<0.25) & (abs(goals[cubes_idx][1]-y)< 0.25) & (z<0.5)


    # axes = [5,5,5]
    # data = np.ones(axes, dtype=np.bool)
    alpha=0.9
    colors = np.empty((10,10,8), dtype=object)
    colors[:] = 'red'

    ax.voxels(x,y,z, cubes,facecolors=colors)
    plt.show()

def midpoints(x):
    sl = ()
    for _ in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x

if __name__=="__main__":
    root_weight = "/home/pinhao/Desktop/keyboard_ws/src/shared_control/stat"
    user_dirs = os.listdir(root_weight)
    k_tr_list = []
    k_g_list = []
    k_tr_p_list = []
    k_g_p_list = []
    distributions = []
    user="paper"
    m = "Trajectron"
    data_dir = os.path.join(root_weight,user,m,'keyboard')
    data_files = os.listdir(data_dir)
    
    for data_file in data_files:
        traj = []
        file_path = os.path.join(data_dir,data_file)
        f = open(file_path,"r")
        if data_file.split("_")[-1] != str(79)+".txt":
            continue
        for line in f:
            # print(line)
            if line.split(":")[0]=="distribution":
                dist = line.split(":")[-1][1:-1].split(" ")
                dist = list(map(float,dist))
                dist = np.array(dist)
                distributions.append(dist)
            if len(distributions) < 2:
                continue
            if line.split(":")[0]=="k_g":
                k_g = line.split(":")[-1][1:-1]
                k_g = float(k_g)
                k_g_list.append(k_g)
            if line.split(":")[0]=="k_tr":
                k_tr = line.split(":")[-1][1:-1]
                k_tr = float(k_tr)
                k_tr_list.append(k_tr)
            if line.split(":")[0]=="k_g_prime":
                k_g_prime = line.split(":")[-1][1:-1]
                k_g_prime = float(k_g_prime)
                k_g_p_list.append(k_g_prime)
            if line.split(":")[0]=="k_tr_prime":
                k_tr_prime = line.split(":")[-1][1:-1]
                k_tr_prime = float(k_tr_prime)
                k_tr_p_list.append(k_tr_prime)
            if line.split(":")[0]=="position_gripper":
                pos = line.split(":")[-1][1:-1].split(" ")
                pos = list(map(float,pos))
                pos = np.array(pos)
                traj.append(pos)

        fig = plt.figure()
        iters = [i for i in range(len(k_g_list))]
        plt.plot(iters, k_g_list, label="k_g", c="red") 
        plt.plot(iters, k_g_p_list, label="k_g'", c="red",linestyle='--')

        plt.plot(iters, k_tr_list, label="k_tr", c="blue") 
        plt.plot(iters, k_tr_p_list, label="k_tr'", c="blue",linestyle='--')
        plt.legend()

        #################### distribution #########################3
        fig = plt.figure()
        distributions = np.stack(distributions,axis=0)
        iters = [i for i in range(distributions.shape[0])]
        for i in range(4):
            plt.plot(iters, distributions[:, 3-i], label="object {}".format(i), c=color[i])
        plt.legend()


        #################### traj #########################
        fig = plt.figure()
        ax = plt.subplot(projection='3d')
        ax.set_xlim(0.2, 0.7)
        ax.set_ylim(-0.3, 0.2)
        ax.set_zlim(0.1, 0.6)

        traj = np.stack(traj,0)
        vis_data = traj
        ax.plot3D(vis_data[:,0], vis_data[ :,1], vis_data[:,2], c='green')
        
        x,y,z = np.indices([22,22,22])/21
        x = x
        y = y - 0.5
        z = z

        xc = midpoints(x)
        yc = midpoints(y)
        zc = midpoints(z)
        
        cubes = []

        cubes.append((abs(goals[0][0]-xc)<=0.025) & (abs(goals[0][1]-yc)<= 0.025) & (zc<=0.05))
        cubes.append((abs(goals[1][0]-xc)<=0.025) & (abs(goals[1][1]-yc)<= 0.025) & (zc<=0.05))
        cubes.append((abs(goals[2][0]-xc)<=0.025) & (abs(goals[2][1]-yc)<= 0.025) & (zc<=0.05))
        cubes.append((abs(goals[3][0]-xc)<=0.025) & (abs(goals[3][1]-yc)<= 0.025) & (zc<=0.05))

        for i in range(len(cubes)):
            cube = cubes[i]
            alpha=0.9
            colors = np.empty(cube.shape, dtype=object)
            colors[:] = color[i]

            ax.voxels(x,y,z, cube,facecolors=colors)

        
        plt.show()
        print()