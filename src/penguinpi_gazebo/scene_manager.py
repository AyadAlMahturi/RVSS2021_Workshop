#!/usr/bin/env python

import sys
import math
import os
import random
from math import pi

import copy
import rospy
import tf
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import DeleteModel, SpawnModel
from gazebo_msgs.srv import SetModelState, GetModelState, GetLinkState
from geometry_msgs.msg import *
from geometry_msgs.msg import Point, Quaternion
import numpy as np
# from msg_converter.pose_msg_converter import *
# from yaml_utils import config_loader
# a Service
# Get model_name
# For every object within the scene shuffle its pose based on engineered poses.
import rospkg

class SceneManager:
    def __init__(self):
        self.obj_class_dict = {
				'aruco1':1, 'aruco2':1, 'aruco3':1, 'aruco4':1, 'aruco5':1, 
				'aruco6':1, 'aruco7':1, 'aruco8':1, 'aruco9':1, 'aruco10':1,
				'apple':1, 'banana':1, 'lemon':1, 'pear':1}
        print(self.obj_class_dict)
        rospack = rospkg.RosPack()
        workspace_path =  rospack.get_path('penguinpi_gazebo')
        urdf_lib_path = workspace_path + '/models/aruco/urdf/'
        self.urdf_dict = {}
        for key in self.obj_class_dict:
            urdf_file_name = urdf_lib_path + key + '.urdf'
            with open(urdf_file_name) as f:
                self.urdf_dict[key] = f.read()
        print(self.urdf_dict.keys())

    def spawn_all_objs(self):
        print("Waiting for gazebo services...")
        rospy.wait_for_service("gazebo/spawn_urdf_model")
        spawn_model = rospy.ServiceProxy("gazebo/spawn_urdf_model", SpawnModel)
        # print("Got it.")
        nx, ny = (8, 8)
        x = np.linspace(-1.2, 1.2, nx)
        y = np.linspace(-1.2, 1.2, ny)
        xv, yv = np.meshgrid(x, y)
        all_candidates = np.arange(nx*ny)
        n_samples = 0
        for key in self.obj_class_dict.keys():
            for _ in range(0, int(self.obj_class_dict[key])):
                n_samples+=1
        rand_idx = random.sample(all_candidates, n_samples)
        obj_counter = 0
        for key in self.obj_class_dict.keys():
            for i in range(0, int(self.obj_class_dict[key])):
                x_idx = int(rand_idx[obj_counter]//nx)
                y_idx = int(rand_idx[obj_counter]%ny)
                x_temp = xv[x_idx, y_idx]
                y_temp = yv[x_idx, y_idx]
                obj_counter += 1
                # x_temp = np.around(np.random.uniform(
                #     -1.5, 1.5, 1), decimals=3)
                # y_temp = np.around(np.random.uniform(
                #     -1.5, 1.5, 1), decimals=3)
                # x_temp = np.around(np.random.uniform(
                #     -4.5, 4.5, 1), decimals=0)
                # y_temp = np.around(np.random.uniform(
                #     -4.5, 4.5, 1), decimals=0)
                item_name = "obj_%s_%i" % (key, i)
                # print("Spawning model:%s", item_name)
                quat = np.array(
                    tf.transformations.quaternion_from_euler(0, 0, 0))
                target_pose = Pose(Point(x_temp, y_temp, 0.06),
                                   Quaternion(quat[0],
                                              quat[1], quat[2], quat[3]))
                spawn_model(item_name, self.urdf_dict[key], "",
                            target_pose, "world")

    def shuffle_objs(self):
        gazebo_model_msg = rospy.wait_for_message('/gazebo/model_states',
                                                  ModelStates)
        model_names = self.get_model_names(gazebo_model_msg)
        for model in model_names:
            if 'mug' in model:
                self.move_n_rot_cup(model)
            else:
                self.move_n_rot_obj(model)

    def shuffle_obj_lazy(self, chosen_objs, hidden_objs):
        for obj in chosen_objs:
            if 'mug' in obj:
                self.move_n_rot_cup(obj)
            else:
                self.move_n_rot_obj(obj)
        for obj in hidden_objs:
            self.hide_obj(obj)

    def get_rand_objs_lazy(self):
        chosen_objs = []
        hidden_objs = []
        for key in self.obj_class_dict:
            if key == 'mug':
                self.rand_obj_dict[key] = np.random.randint(
                    1, self.obj_class_dict[key] + 1, size=1)
            else:
                self.rand_obj_dict[key] = np.random.randint(
                    self.obj_class_dict[key] + 1, size=1)
        for key in self.rand_obj_dict:
            for i in range(0, self.rand_obj_dict[key]):
                name_temp = ['obj_' + key + '_' + str(i)]
                chosen_objs += name_temp
            num_hidden_obj = self.obj_class_dict[key] - self.rand_obj_dict[key]
            if num_hidden_obj > 0:
                for i in range(self.rand_obj_dict[key],
                               self.obj_class_dict[key]):
                    name_temp = ['obj_' + key+'_' + str(i)]
                    hidden_objs += name_temp
        return chosen_objs, hidden_objs

    def delete_all_objs(self):
        for key in self.obj_class_dict:
            for i in range(0, int(self.obj_class_dict[key])):
                obj_name = "obj_%s_%i" % (key, i)
                # print("deleting model:%s", obj_name)
                self.delete_model(obj_name)
                rospy.sleep(0.05)

    def delete_model(self, model_name):
        try:
            delete_model = rospy.ServiceProxy("gazebo/delete_model",
                                              DeleteModel)
            delete_model(model_name)
        except rospy.ServiceException, e:
            print "Service call failed: %s"

    #################################
    #           Tool Box            #
    #################################

    def move_obj(self, target):
        try:
            set_cube_position = rospy.ServiceProxy(
                '/gazebo/set_model_state', SetModelState)
            move_status = set_cube_position(target)
        except rospy.ServiceException, e:
            print "Service call failed: %s"

    def hide_obj(self, obj_name):
        new_quat = tf.transformations.quaternion_from_matrix(np.eye(4))
        x, y, z = 0.0, -1.5, 0.1
        # x = np.random.choice([-0.3, 0.3], 1)
        # y = np.random.choice([-0.15, 0.25], 1)
        target = ModelState()
        target.model_name = obj_name
        target.reference_frame = "link"
        target.pose.position = Point(x, y, z)
        target.pose.orientation = Quaternion(new_quat[0], new_quat[1],
                                             new_quat[2], new_quat[3])
        self.move_obj(target)

    def get_model_names(self, model_states_msg):
        model_names = model_states_msg.name
        obj_names = []
        for model_name in model_names:
            if 'obj' in model_name:
                obj_names.append(model_name)
        return obj_names

    # # combine get_model_names,
    # def get_obj_abbres_n_poses(self, model_states_msg, chosen_objs):
    #     all_model_names = model_states_msg.name
    #     all_model_poses_quat = model_states_msg.pose
    #     chosen_obj_abbres = []
    #     chosen_obj_poses = np.zeros((7, 1))
    #     for idx, obj in enumerate(chosen_objs):
    #         target_idx = all_model_names.index(obj)
    #         obj_pose_quat = pose_msg2vec(all_model_poses_quat[target_idx])
    #         # the object is on the table
    #         obj_x = obj_pose_quat[0]
    #         obj_y = obj_pose_quat[1]
    #         obj_z = obj_pose_quat[2]
    #         if abs(obj_x)<2 and abs(obj_y) < 1 and 0.7<obj_z<2:
    #             chosen_obj_poses = np.hstack((chosen_obj_poses, obj_pose_quat.
    #                                           reshape((7, 1))))
    #             obj_abbre = self.get_obj_abbre(obj)
    #             chosen_obj_abbres.append(obj_abbre)
    #     # print obj_counter
    #     return chosen_obj_abbres, chosen_obj_poses[:, 1:]

    # def get_obj_abbre(self, obj_name):
    #     split_name = obj_name.split('_')
    #     class_acro = split_name[1]
    #     obj_idx = class_acro + '_' + split_name[2]
    #     return obj_idx

    def get_link_pose(self, link_name, reference_frame, repr='se3'):
        rospy.wait_for_service('/gazebo/get_link_state')
        try:
            current_link_state = rospy.ServiceProxy(
                '/gazebo/get_link_state', GetLinkState)
            link_state = current_link_state(link_name, reference_frame)
            link_pose = link_state.link_state
            if repr == 'se3':
                return pose_msg2se3(link_pose.pose)
            elif repr == 'quat':
                return pose_msg2vec(link_pose.pose)
        except rospy.ServiceException, e:
            print ("Service call failed: %s")

    def get_model_pose(self, model_name, reference_frame, repr='se3'):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            current_link_state = rospy.ServiceProxy(
                '/gazebo/get_model_state', GetModelState)
            link_state = current_link_state(model_name, reference_frame)
            if repr == 'se3':
                return pose_msg2se3(link_state.pose)
            elif repr == 'quat':
                return pose_msg2vec(link_state.pose)
        except rospy.ServiceException, e:
            print
            "Service call failed: %s"

    def rot_mat(self, angle, axis):
        c = math.cos(angle)
        s = math.sin(angle)
        if axis == 'x':
            rot_mat = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == 'y':
            rot_mat = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif axis == 'z':
            rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            print("####### Warning !!! #######")
            print(' chose between x, y, z axis ')
            print("###########################")
            rospy.signal_shutdown('Quit')
        return rot_mat

    def get_rand_objs(self):
        for key in self.obj_class_dict:
            self.rand_obj_dict[key] = np.random.randint(
                self.obj_class_dict[key] + 1, size=1)




def main():
    rospy.init_node("spawn_objects")
    obj_manager = SceneManager()
    obj_manager.spawn_all_objs()


if __name__ == '__main__':
    main()
