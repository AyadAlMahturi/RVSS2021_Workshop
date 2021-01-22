import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import matplotlib.patches as label_box
import cv2 
import os, sys
import time
# Import integration components
sys.path.insert(0, "{}/integration".format(os.getcwd()))
import integration.penguinPiC
import integration.DatasetHandler as dh
import control.keyboardControl as Keyboard

# Import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
import slam.Slam_cv2 as Slam
import slam.Robot as Robot
import slam.aruco_detector as aruco
import slam.Measurements as Measurements

# Import network components
sys.path.insert(0,"{}/network".format(os.getcwd()))
from network.detector import Detector
from PIL import Image
import io

plt.ion()

class Operate:
    def __init__(self, datadir, ppi, writeData=False):
        # Initialise data parameters
        self.ppi = ppi
        self.ppi.set_velocity(0,0)
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        ckpt = "network/weights/pretrained_backbone/pretrained_backbone.pth.tar"
        self.detector = Detector(ckpt,use_gpu=False)
        self.colour_map = np.zeros([240,320,3], dtype=np.uint8)
        self.image_display = None
        self.inference_display = None

        # Set up subsystems
        camera_matrix, dist_coeffs, scale, baseline = self.getCalibParams(datadir)

        # Control subsystem
        self.keyboard = Keyboard.Keyboard(self.ppi)
        # SLAM subsystem
        self.pibot = Robot.Robot(baseline, scale, camera_matrix, dist_coeffs)
        self.aruco_det = aruco.aruco_detector(self.pibot, marker_length = 0.07)
        self.slam = Slam.Slam(self.pibot)
        
        # Optionally record input data to a dataset
        if writeData:
            self.data = dh.DatasetWriter('test')
        else:
            self.data = None
        self.output = dh.OutputWriter('system_output')
        #

        # gui canvas
        self.robot_view = plt.subplot(221)
        self.infernce_view = plt.subplot(223)
        self.slam_view = plt.subplot(122)
        # TODO: reduce legend size
        self.timer = time.time()
        self.canvas = None


    def __del__(self):
        self.ppi.set_velocity(0,0)

    def getCalibParams(self,datadir):
        # Imports calibration parameters
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')

        return camera_matrix, dist_coeffs, scale, baseline

    def control(self):
        lv, rv = self.keyboard.latest_drive_signal()
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.timer
        drive_meas = Measurements.DriveMeasurement(lv, rv, dt)
        self.slam.predict(drive_meas)
        self.timer = time.time()

    def take_pic(self):
        self.img = self.ppi.get_image()
        if not self.data is None:
            self.data.write_image(self.img)
    
    def update_slam(self):
        lms, _ = self.aruco_det.detect_marker_positions(self.img)       
        self.slam.add_landmarks(lms)
        self.slam.update(lms)

    def detect_fruit(self): 
        if self.keyboard.get_net_signal():
            pred, self.colour_map = self.detector.detect_single_image(self.img)
            # print(np.unique(pred))
            return pred
        else:
            # TODO: change output to zeros
            return None

    def update_gui(self):
        font = cv2.FONT_HERSHEY_SIMPLEX 

        pad = 40
        bg_rgb = np.array([79, 106, 143]).reshape(1, 1, 3)
        if self.canvas is None:
            self.canvas = np.ones((480+3*pad, 640+3*pad, 3))*bg_rgb
        self.canvas = self.canvas.astype(np.uint8)

        slam_plot = self.slam.draw_slam_state(res=(320, 480+pad))
        labels = ["apple", 'banana', 'pear', 'lemon']
        legend_h = 10
        legend_w = 20
        off_set = 10
        pt1 = (260, 10)

        self.canvas[pad:480+2*pad, 2*pad+320:2*pad+2*320, :] = slam_plot
        self.canvas[pad:240+pad, pad:320+pad, :] =  cv2.resize(self.img, (320, 240))
        if self.colour_map is not None:
            for i in range(4):
                c = self.detector.colour_code[i]
                self.colour_map = cv2.rectangle(self.colour_map, pt1, (pt1[0]+legend_w, pt1[1]+legend_h),
                    color = (int(c[0]), int(c[1]), int(c[2])), thickness=-1)
                self.colour_map = cv2.putText(self.colour_map, labels[i], (pt1[0]+legend_w+off_set, pt1[1]),
                    font, 0.2, (0, 30, 56), 1, cv2.LINE_AA)
                pt1 = (pt1[0], pt1[1]+legend_h+off_set)
            self.canvas[(240+2*pad):(240+2*pad+240), pad:(320+pad), :] = \
                  cv2.resize(self.colour_map, (320, 240), cv2.INTER_NEAREST)
        out = cv2.cvtColor(self.canvas.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow('In the Jungle', out)
        cv2.waitKey(1)

    def record_data(self):
        # Save data for network processing
        if self.keyboard.get_net_signal():
            self.output.write_image(self.img, self.slam)
        if self.keyboard.get_slam_signal():
            self.output.write_map(self.slam)


if __name__ == "__main__":   
    # plt.ion()
    currentDir = os.getcwd()
    datadir = "{}/Calibration/param/".format(currentDir)
    # Use either a real or simulated penguinpi
    #ppi = integration.penguinPiC.PenguinPi(ip = '192.168.50.1')
    ppi = integration.penguinPiC.PenguinPi()    
    # ppi = dh.DatasetPlayer("test")
    # Set up the integrated system
    operate = Operate(datadir, ppi, writeData=False)
    # Enter the main loop
    img_buffer = None
    while 1:
        operate.control()
        #tick = time.time()
        operate.take_pic()
        # if img_buffer is None:
        #     img_buffer = operate.img
        # else:
        #     print(np.sum(img_buffer-operate.img))
        #     img_buffer = operate.img
        # print(f'{1/(time.time()-tick):.2f} FPS take_pic')
        # print(f'{1/(time.time()-tick):.2f} FPS')
        # tick = time.time()
        operate.update_slam()
        #print(f'{1/(time.time()-tick):.2f} FPS slam')
        #tick = time.time()
        operate.detect_fruit()
        #print(f'{1/(time.time()-tick):.2f} FPS detect')
        tick = time.time()
        operate.update_gui()
        print(f'{1/(time.time()-tick):.2f} FPS Plt')




