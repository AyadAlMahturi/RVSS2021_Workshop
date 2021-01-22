import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

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
import slam.Slam as Slam
import slam.Robot as Robot
import slam.aruco_detector as aruco
import slam.Measurements as Measurements

# Import network components
sys.path.insert(0,"{}/network".format(os.getcwd()))
from network.detector import Detector
from PIL import Image
import io

# plt.ion()

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
            print(np.unique(pred))
            return pred
        else:
            # TODO: change output to zeros
            return None

    def update_gui(self):
        
        # Output system to screen
        self.slam_view.cla()
        self.slam.draw_slam_state(self.slam_view)
        # self.slam.draw_slam_state(self.slam_view)
        # plt.show()
        # cv2.imshow('test', canvas)
        # cv2.waitKey(1)

        
        if self.image_display is None:
            self.image_display = self.robot_view.imshow(self.img)
            self.robot_view.axis('off')
            self.robot_view.set_title('PiBot Cam View')
        else:
            self.image_display.set_data(self.img)
        plt.pause(0.001)

        if self.inference_display is None:
            self.inference_display = self.infernce_view.imshow(self.colour_map, interpolation = 'nearest')
            self.infernce_view.axis('off')
            apple_label = label_box.Patch(color=self.detector.colour_code[1]/255, label='apple[1]')
            banana_label = label_box.Patch(color=self.detector.colour_code[2]/255, label='banana[2]')
            pear_label = label_box.Patch(color=self.detector.colour_code[3]/255, label='pear[3]')
            lemon_label = label_box.Patch(color=self.detector.colour_code[4]/255, label='lemon[4]')
            self.infernce_view.legend(
                handles=[apple_label, banana_label, pear_label, lemon_label], prop={'size':4},loc=4)
            self.infernce_view.set_title("Fruit Detection")
        else:
            self.inference_display.set_data(self.colour_map)

        # self.infernce_view.cla()
        # self.infernce_view.imshow(self.colour_map,
        #  interpolation='nearest')        
        # self.infernce_view.axis('off')
        # apple_label = label_box.Patch(color=self.detector.colour_code[1]/255, label='apple[1]')
        # banana_label = label_box.Patch(color=self.detector.colour_code[2]/255, label='banana[2]')
        # pear_label = label_box.Patch(color=self.detector.colour_code[3]/255, label='pear[3]')
        # lemon_label = label_box.Patch(color=self.detector.colour_code[4]/255, label='lemon[4]')
        # self.infernce_view.legend(
        #     handles=[apple_label, banana_label, pear_label, lemon_label], prop={'size':4},loc=4)
        # self.infernce_view.set_title("Fruit Detection")

        # # plt.tight_layout()
        # plt.pause(0.001)
        # buf = io.BytesIO()
        # plt.savefig(buf, format='png')
        # buf.seek(0)
        # im = Image.open(buf)
        # im = np.asarray(im)
        # # im.show()
        # buf.close()

  
        # cv2.waitKey(5)


        
        
    
    def record_data(self):
        # Save data for network processing
        if self.keyboard.get_net_signal():
            self.output.write_image(self.img, self.slam)
        if self.keyboard.get_slam_signal():
            self.output.write_map(self.slam)
    
    def animate_image(self, n, img_disp):
        self.take_pic()
        img_disp.set_data(self.img)
        return img_disp


if __name__ == "__main__":   
    plt.ion()
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
        tick = time.time()
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
        # tick = time.time()
        operate.update_gui()
        # plt.pause(1e-3)
        print(f'{1/(time.time()-tick):.2f} FPS Plt')




