import numpy as np
import cv2 
import os, sys
import time
# Import integration components
sys.path.insert(0, "{}/integration".format(os.getcwd()))
from control.pibot import PenguinPi
import integration.DatasetHandler as dh
# import control.keyboardControl as Keyboard
import pygame

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
    def __init__(self, args):
        # Initialise data parameters
        
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)
            
        self.pred = np.zeros([240,320,3], dtype=np.uint8)
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        ckpt = "network/scripts/res18_skip_weights.pth"
        # ckpt = ""
        if ckpt == "":
            self.detector = None
        else:
            self.detector = Detector(ckpt, use_gpu=False)
        self.colour_map = np.ones([240,320,3], dtype=np.uint8)
        self.colour_map *= 100
        # Set up subsystems
        camera_matrix, dist_coeffs, scale, baseline = self.getCalibParams(
            args.calib_dir, args.ip)

        # Control subsystem
        # SLAM subsystem
        self.robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        self.aruco_det = aruco.aruco_detector(self.robot, marker_length = 0.07)
        self.slam = EKF(self.robot)
        
        # Optionally record input data to a dataset
        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None

        self.output = dh.OutputWriter('workshop_output')
        self.timer = time.time()
        self.count_down = 180
        self.start_time = time.time()
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False}
        self.close = False
        self.pred_fname = ''
        self.recover_slam = False
        self.notification = 'Press ENTER to start SLAM'
        self.output_state = None
        self.slam_on = False
        # self.debug_flag = False

    def getCalibParams(self, datadir, ip):
        # Imports calibration parameters
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')
        return camera_matrix, dist_coeffs, scale, baseline

    def control(self):       
        if args.play_data:
            lv, rv = self.pibot.set_velocity()            
        else:
            motion_command = self.command['motion']
            lv, rv = self.pibot.set_velocity(motion_command)
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.timer
        drive_meas = measure.Drive(lv, rv, dt)
        self.timer = time.time()
        return drive_meas

    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)
       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.recover_slam:
            is_success = self.slam.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.recover_slam = False
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.recover_slam = True
                self.slam_on = False
        elif self.slam_on: # and not self.debug_flag:
            self.slam.predict(drive_meas)
            self.slam.add_landmarks(lms)
            self.slam.update(lms)

    def detect_fruit(self):
        if self.detector is None:
            warning = "No valid Checkpoint"
            cv2.putText(self.colour_map, warning, (40, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), thickness=2)
        elif self.command['inference'] and self.detector is not None:
            self.pred, self.colour_map = self.detector.detect_single_image(self.img)
            self.command['inference'] = False
            self.output_state = (self.pred, self.slam)
            self.notification = f'{len(np.unique(self.pred))-1} fruit type(s) detected'

    def draw(self):        
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
        # slam view
        if self.slam_on:
            canvas[pad:480+2*pad, 2*pad+320:2*pad+2*320, :] = \
                self.slam.draw_slam_state(res=(320, 480+pad))
        else:
            canvas[pad:480+2*pad, 2*pad+320:2*pad+2*320, :] = \
                self.slam.draw_slam_state(res=(320, 480+pad))/2

        # robot view
        canvas[pad:240+pad, pad:320+pad, :] = \
             cv2.resize(self.aruco_img, (320, 240))
        # prediction view
        canvas[(240+2*pad):(240+2*pad+240), pad:(320+pad), :] = \
                cv2.resize(self.colour_map, (320, 240), cv2.INTER_NEAREST)
        # out = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_RGB2BGR)
        return canvas

    def update_keyboard(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'][0] = min(self.command['motion'][0]+1, 1)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'][0] = max(self.command['motion'][0]-1, -1)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'][1] = min(self.command['motion'][1]+1, 1)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'][1] = max(self.command['motion'][1]-1, -1)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.command['reset_slam'] = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_markers = len(self.slam.taglist)
                if n_markers == 0:
                    self.slam_on = True
                    self.notification = 'SLAM is running'
                elif n_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.slam_on:
                        self.recover_slam = True
                    self.slam_on = not self.slam_on
            if event.type == pygame.QUIT:
                self.close = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close = True
        if self.close:
            pygame.quit()
            sys.exit()

    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.slam)
            self.notification = 'Map is saved'
            self.command['output'] = False
        if self.command['save_inference']:
            if self.output_state is not None:
                self.pred_fname = self.output.write_image(self.output_state[0],
                                                        self.output_state[1])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False
        

        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    args, _ = parser.parse_known_args()
    
    pygame.font.init() 
    
    width, height = 760, 600
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('RVSS 2021 Workshop')
    icon = pygame.image.load('pics/logo.png')
    pygame.display.set_icon(icon)
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/rvss_splash.png')
    logo = pygame.image.load('pics/logo_small.png')
    # splash = pygame.transform.scale(splash, (width, height))
    pygame.display.update()

    # print(f'Use the arrow keys to drive the robot.')
    # print('Press P to detect the fruit.')
    # print('Press S to record the SLAM map.')
    # print('Press N to record the inference and robot position.')

    start = False

    counter = 0
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 700)
        canvas.blit(logo, (x_, 500))
        pygame.display.update()
        counter += 1

    pygame_font = pygame.font.SysFont('Comic Sans MS', 30)

    operate = Operate(args)

    while start:
        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.detect_fruit()
        # visualise
        img_surface = pygame.surfarray.make_surface(operate.draw())
        img_surface = pygame.transform.flip(img_surface, True, False)
        img_surface = pygame.transform.rotozoom(img_surface, 90, 1)
        canvas.blit(img_surface, (0, 0))

        text_surface = pygame_font.render(operate.notification, False, (200, 200, 200))
        canvas.blit(text_surface, (40, 570))

        time_remain = operate.count_down - time.time() + operate.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = pygame_font.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (470, 60))
        #
        pygame.display.update()




