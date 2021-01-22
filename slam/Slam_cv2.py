import numpy as np
import SlamMap
import matplotlib.patches as patches
import cv2

class Slam:
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
        state = np.concatenate((self.robot.state, np.reshape(self.markers, (-1,1), order='F')), axis=0)
        return state
    
    def set_state_vector(self, state):
        self.robot.state = state[0:3,:]
        self.markers = np.reshape(state[3:,:], (2,-1), order='F')
    
    def save_map(self, fname="slam_map.txt"):
        if self.number_landmarks() > 0:
            slam_map = SlamMap.SlamMap(self.markers, self.P[3:,3:], self.taglist)
            slam_map.save(fname)
        
    
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
        robot_xy = robot_xy - self.robot_init_state
        lms_xy = lms_xy - self.robot_init_state
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
        # draw landmards
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                canvas = cv2.circle(canvas, coor_, radius=3, color=(155, 5, 23), thickness=4)
                # plot covariance
                lmi = self.markers[:,i]
                Plmi = self.P[3+2*i:3+2*(i+1),3+2*i:3+2*(i+1)]
                axes_len, angle = self.make_ellipse(lmi, Plmi)
                canvas = cv2.ellipse(canvas, coor_, (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)),
                    angle, 0, 360, (244, 69, 96), 1)
                # print text
                text_coor_ = (coor_[0]+5, coor_[1])
                cv2.putText(canvas, f'{self.taglist[i]}', text_coor_, font, 0.5, (0, 30, 56), 1, cv2.LINE_AA)
        return canvas
    
    @staticmethod
    def make_ellipse(x, P):
        e_vals, e_vecs = np.linalg.eig(P)
        idx = e_vals.argsort()[::-1]   
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        alpha = np.sqrt(4.605)
        axes_len = e_vals*2*alpha
        # print(axes_len)
        angle = np.arctan(e_vecs[0, 0]/e_vecs[1, 0])
        return (axes_len[0], axes_len[1]), angle

        # t = np.linspace(0, 2 * np.pi)
        # ellipse = (e_vecs @ np.sqrt(np.diag(e_vals))) @ np.block([[np.cos(t)],[np.sin(t)]])
        # ellipse = ellipse + x.reshape(-1,1)

        # return ellipse
