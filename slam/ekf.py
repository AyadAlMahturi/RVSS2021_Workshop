import numpy as np
from mapping_utils import MappingUtils
import cv2
import math

class EKF:
    # Implementation of an EKF for SLAM
    # The state is ordered as [x; y; theta; l1x; l1y; ...; lnx; lny]

    # Utility
    # -------

    def __init__(self, robot):
        # State components
        self.robot = robot
        self.markers = np.zeros((2,0))
        self.taglist = []

        # Covariance matrix
        self.P = np.zeros((3,3))
        self.init_lm_cov = 1e3
        self.robot_init_state = None

    def number_landmarks(self):
        return int(self.markers.shape[1])

    def get_state_vector(self):
        state = np.concatenate(
            (self.robot.state, np.reshape(self.markers, (-1,1), order='F')), axis=0)
        return state
    
    def set_state_vector(self, state):
        self.robot.state = state[0:3,:]
        self.markers = np.reshape(state[3:,:], (2,-1), order='F')
    
    def save_map(self, fname="slam_map.txt"):
        if self.number_landmarks() > 0:
            utils = MappingUtils(self.markers, self.P[3:,3:], self.taglist)
            utils.save(fname)
        
    
    # EKF functions
    # -------------

    def predict(self, raw_drive_meas):
        # The prediction step of EKF

        F = self.state_transition(raw_drive_meas)
        x = self.get_state_vector()

        Q = self.predict_covariance(raw_drive_meas)

        self.robot.drive(raw_drive_meas)
        x[0:3, :] = self.robot.state
        self.P = F @ self.P @ F.T + Q

        self.set_state_vector(x)


    def update(self, measurements):
        if not measurements:
            return

        # Construct measurement index list
        tags = [lm.tag for lm in measurements]
        idx_list = [self.taglist.index(tag) for tag in tags]

        # Stack measurements and set covariance
        z = np.concatenate([lm.position.reshape(-1,1) for lm in measurements], axis=0)
        R = np.zeros((2*len(measurements),2*len(measurements)))
        for i in range(len(measurements)):
            R[2*i:2*i+2,2*i:2*i+2] = measurements[i].covariance


        # Compute own measurements
        z_hat = self.robot.measure(self.markers, idx_list)
        z_hat = z_hat.reshape((-1,1),order="F")
        H = self.robot.derivative_measure(self.markers, idx_list)

        x = self.get_state_vector()

        y = z - z_hat
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        x = x + K @ y
        self.P = (np.eye(x.shape[0]) - K @ H) @ self.P
        self.set_state_vector(x)

        # print(self.P)
        # print("MarkerMeasurement residual:\n", y)
    
    def recover_from_pause(self,measurements):
        if not measurements:
            return False
        


        lm_new = np.zeros((2,0))
        lm_prev = np.zeros((2,0))
        tag = []
        for lm in measurements:
            if lm.tag in self.taglist:
                lm_new = np.concatenate((lm_new, lm.position), axis=1)
                tag.append(int(lm.tag))
                lm_idx = np.where(self.taglist == lm.tag)[0][0]
                lm_prev = np.concatenate((lm_prev,self.markers[:,lm_idx].reshape(2, 1)), axis=1)
        
        if int(lm_new.shape[1])<2:
            return False

        R,t = self.umeyama(lm_new,lm_prev)
        print(R)
        print(t)
        theta = math.atan2(R[0][0],R[1][0])
        self.robot.state[0]=t[0]
        self.robot.state[1]=t[1]
        self.robot.state[2]=theta
        return True




    def state_transition(self, raw_drive_meas):
        n = self.number_landmarks()*2 + 3
        F = np.eye(n)
        F[0:3,0:3] = self.robot.derivative_drive(raw_drive_meas)
        return F
    
    def predict_covariance(self, raw_drive_meas):
        n = self.number_landmarks()*2 + 3
        Q = np.zeros((n,n))
        Q[0:3,0:3] = self.robot.covariance_drive(raw_drive_meas)+ 0.01*np.eye(3)
        return Q

    def add_landmarks(self, measurements):
        if not measurements:
            return

        th = self.robot.state[2]
        robot_xy = self.robot.state[0:2,:]
        R_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])

        # Add new landmarks to the state
        for lm in measurements:
            if lm.tag in self.taglist:
                # ignore known tags
                continue
            
            lm_bff = lm.position
            lm_inertial = robot_xy + R_theta @ lm_bff

            self.taglist.append(int(lm.tag))
            self.markers = np.concatenate((self.markers, lm_inertial), axis=1)

            # Create a simple, large covariance to be fixed by the update step
            self.P = np.concatenate((self.P, np.zeros((2, self.P.shape[1]))), axis=0)
            self.P = np.concatenate((self.P, np.zeros((self.P.shape[0], 2))), axis=1)
            self.P[-2,-2] = self.init_lm_cov**2
            self.P[-1,-1] = self.init_lm_cov**2

    # Plotting functions
    # ------------------
    @ staticmethod
    def to_im_coor(xy, res, m2pixel):
        w, h = res
        x, y = xy
        x_im = int(-x*m2pixel+w/2.0)
        y_im = int(y*m2pixel+h/2.0)
        return (x_im, y_im)

    def draw_slam_state(self, res = (320, 500)):
        font = cv2.FONT_HERSHEY_SIMPLEX 
        # Draw landmarks
        m2pixel = 80
        bg_rgb = np.array([213, 213, 213]).reshape(1, 1, 3)
        canvas = np.ones((res[1], res[0], 3))*bg_rgb.astype(np.uint8)
        # in meters, 
        lms_xy = self.markers[:2, :]
        robot_xy = self.robot.state[:2, 0].reshape((2, 1))
        if self.robot_init_state is None:
            self.robot_init_state = robot_xy
        lms_xy = lms_xy - robot_xy
        robot_xy = robot_xy - robot_xy

        # plot robot
        arrow_scale = 0.3
        end_point_x = arrow_scale*np.cos(self.robot.state[2,0]) + robot_xy[0, 0]
        end_point_y =  arrow_scale*np.sin(self.robot.state[2,0]) + robot_xy[1, 0]
        end_point = (end_point_x, end_point_y)
        start_point = (robot_xy[0,0], robot_xy[1,0]) 
        start_point_uv = self.to_im_coor(start_point, res, m2pixel)
        end_point_uv = self.to_im_coor(end_point, res, m2pixel)
        canvas = cv2.circle(canvas, start_point_uv , radius=3, color=(0, 30, 56), thickness=4)
        canvas = cv2.arrowedLine(canvas,start_point_uv,  
            end_point_uv, color=(0, 30, 56), thickness=2, tipLength=0.5) 
        
        p_robot = self.P[0:2,0:2]
        axes_len,angle = self.make_ellipse(p_robot)
        canvas = cv2.ellipse(canvas, start_point_uv, 
                    (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)),
                    angle, 0, 360, (244, 69, 96), 1)
        # draw landmards
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                canvas = cv2.circle(canvas, coor_, radius=3, color=(155, 5, 23), thickness=4)
                # plot covariance

                lmi = self.markers[:,i]
                Plmi = self.P[3+2*i:3+2*(i+1),3+2*i:3+2*(i+1)]
                axes_len, angle = self.make_ellipse(Plmi)
                canvas = cv2.ellipse(canvas, coor_, 
                    (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)),
                    angle, 0, 360, (244, 69, 96), 1)
                # print text
                text_coor_ = (coor_[0]+5, coor_[1])
                cv2.putText(canvas, f'{self.taglist[i]}',
                 text_coor_, font, 0.5, (0, 30, 56), 1, cv2.LINE_AA)
        return canvas
    
    @staticmethod
    def make_ellipse(P):
        e_vals, e_vecs = np.linalg.eig(P)
        idx = e_vals.argsort()[::-1]   
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        alpha = np.sqrt(4.605)
        axes_len = e_vals*2*alpha
        angle = np.arctan(e_vecs[0, 0]/e_vecs[1, 0])
        return (axes_len[0], axes_len[1]), angle

    @staticmethod
    def umeyama(from_points, to_points):

    
        assert len(from_points.shape) == 2, \
            "from_points must be a m x n array"
        assert from_points.shape == to_points.shape, \
            "from_points and to_points must have the same shape"
        
        N = from_points.shape[1]
        m = 2
        
        mean_from = from_points.mean(axis = 1).reshape((2,1))
        mean_to = to_points.mean(axis = 1).reshape((2,1))
        
        delta_from = from_points - mean_from # N x m
        delta_to = to_points - mean_to       # N x m
        
        sigma_from = (delta_from * delta_from).sum(axis = 1).mean()
        sigma_to = (delta_to * delta_to).sum(axis = 1).mean()
        
        cov_matrix = delta_to @ delta_from.T / N
        
        U, d, V_t = np.linalg.svd(cov_matrix, full_matrices = True)
        cov_rank = np.linalg.matrix_rank(cov_matrix)
        S = np.eye(m)
        
        if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
            S[m-1, m-1] = -1
        elif cov_rank < m-1:
            raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))
        
        R = U.dot(S).dot(V_t)
        # c = (d * S.diagonal()).sum() / sigma_from
        t = mean_to - R.dot(mean_from)
    
        return R, t
    
    @staticmethod
    def umeyama_alignment(x, y, with_scale = False):

        if x.shape != y.shape:
            raise GeometryException("data matrices must have the same shape")

        # m = dimension, n = nr. of data points
        m, n = x.shape

        # means, eq. 34 and 35
        mean_x = x.mean(axis=1)
        mean_y = y.mean(axis=1)

        # variance, eq. 36
        # "transpose" for column subtraction
        sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

        # covariance matrix, eq. 38
        outer_sum = np.zeros((m, m))
        for i in range(n):
            outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
        cov_xy = np.multiply(1.0 / n, outer_sum)

        # SVD (text betw. eq. 38 and 39)
        u, d, v = np.linalg.svd(cov_xy)
        if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
            raise GeometryException("Degenerate covariance rank, "
                                    "Umeyama alignment is not possible")

        # S matrix, eq. 43
        s = np.eye(m)
        if np.linalg.det(u) * np.linalg.det(v) < 0.0:
            # Ensure a RHS coordinate system (Kabsch algorithm).
            s[m - 1, m - 1] = -1

        # rotation, eq. 40
        r = u.dot(s).dot(v)

        # scale & translation, eq. 42 and 41
        c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
        t = mean_y - np.multiply(c, r.dot(mean_x))

        return r, t, c
