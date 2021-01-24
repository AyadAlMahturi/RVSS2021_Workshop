import numpy as np
import cv2 
import os, sys
import time
# Import integration components
sys.path.insert(0, "{}/integration".format(os.getcwd()))
from control.pibot_sim import PenguinPi
import integration.DatasetHandler as dh
import control.keyboardControl as Keyboard

# Import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco
import slam.measure as measure

# Import network components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))

from network.scripts.detector import Detector


class Operate:
    def __init__(self, datadir, ppi, writeData=False):
        # Initialise data parameters
        self.ppi = ppi
        self.ppi.set_velocity(0,0)
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        ckpt = "network/scripts/weights/pretrained_backbone/pretrained_backbone.pth.tar"
        self.detector = Detector(ckpt,use_gpu=False)
        self.colour_map = np.zeros([240,320,3], dtype=np.uint8)
        self.image_display = None
        self.inference_display = None

        # Set up subsystems
        camera_matrix, dist_coeffs, scale, baseline = self.getCalibParams(datadir)

        # Control subsystem
        self.keyboard = Keyboard.Keyboard(self.ppi)
        # SLAM subsystem
        self.pibot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        self.aruco_det = aruco.aruco_detector(self.pibot, marker_length = 0.07)
        self.slam = EKF(self.pibot)
        
        # Optionally record input data to a dataset
        if writeData:
            self.data = dh.DatasetWriter('test')
        else:
            self.data = None
        self.output = dh.OutputWriter('system_output')
        #

        # TODO: reduce legend size
        self.timer = time.time()
        self.count_down = 180
        self.start_time = time.time()


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
        drive_meas = measure.Drive(lv, rv, dt)
        self.slam.predict(drive_meas)
        self.timer = time.time()

    def take_pic(self):
        self.img = self.ppi.get_image()
        if not self.data is None:
            self.data.write_image(self.img)
    
    def update_slam(self):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)       
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
        pad = 40
        bg_rgb = np.array([79, 106, 143]).reshape(1, 1, 3)
        canvas = np.ones((480+3*pad, 640+3*pad, 3))*bg_rgb
        canvas = canvas.astype(np.uint8)
        # 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(canvas, "PiBot Cam", (138, 31),
                    font, 0.8, (200, 200, 200), thickness=2)
        cv2.putText(canvas, "SLAM Map", (492, 31),
                    font, 0.8, (200, 200, 200), thickness=2)
        cv2.putText(canvas, "Fruit Detection", (110, 310),
                    font, 0.8, (200, 200, 200), thickness=2)
        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        cv2.putText(canvas, time_remain, (490, 590),
                    font, 0.8, (200, 200, 200), thickness=2)
        # slam view
        canvas[pad:480+2*pad, 2*pad+320:2*pad+2*320, :] = \
            self.slam.draw_slam_state(res=(320, 480+pad))
        # robot view
        canvas[pad:240+pad, pad:320+pad, :] = \
             cv2.resize(self.aruco_img, (320, 240))
        # prediction view
        canvas[(240+2*pad):(240+2*pad+240), pad:(320+pad), :] = \
                cv2.resize(self.colour_map, (320, 240), cv2.INTER_NEAREST)
        out = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow('In the Jungle', out)
        cv2.waitKey(1)

    def record_data(self):
        # Save data for network processing
        if self.keyboard.get_net_signal():
            self.output.write_image(self.img, self.slam)
        if self.keyboard.get_slam_signal():
            self.output.write_map(self.slam)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    args, _ = parser.parse_known_args()

    currentDir = os.getcwd()
    datadir = "{}/calibration/param/".format(currentDir)
    # Use either a real or simulated penguinpi
    #ppi = integration.penguinPiC.PenguinPi(ip = '192.168.50.1')
    print(args.ip, args.port)
    ppi = PenguinPi(args.ip, args.port)
    # ppi = dh.DatasetPlayer("test")
    # Set up the integrated system
    operate = Operate(datadir, ppi, writeData=False)
    # Enter the main loop
    while 1:
        operate.control()
        # tick = time.time()
        operate.take_pic()
        operate.update_slam()
        # print(f'{1/(time.time()-tick):.2f} FPS slam')
        # tick = time.time()
        operate.detect_fruit()
        # print(f'{1/(time.time()-tick):.2f} FPS detect')
        # tick = time.time()
        operate.update_gui()
        # print(f'{1/(time.time()-tick):.2f} FPS Plt')




