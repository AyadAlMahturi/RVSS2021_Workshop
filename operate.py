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
        ckpt = "network/scripts/res18_skip_weights.pth"
        # ckpt = ""
        if ckpt == "":
            self.detector = None
        else:
            self.detector = Detector(ckpt, use_gpu=False)
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(self.ekf.robot, marker_length = 0.07)
        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('workshop_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.notification = 'Press ENTER to start SLAM'
        #
        self.count_down = 180
        self.start_time = time.time()
        self.control_clock = time.time()
        #
        self.detector_output = np.zeros([240,320], dtype=np.uint8)
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.network_vis = np.ones([240,320,3], dtype=np.uint8) * 100
        self.gui_mask = pygame.image.load('pics/gui_mask.png')


    def control(self):       
        if args.play_data:
            lv, rv = self.pibot.set_velocity()            
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'])
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        drive_meas = measure.Drive(lv, rv, dt)
        self.control_clock = time.time()
        return drive_meas

    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)
       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.request_recover_robot = False
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.request_recover_robot = True
                self.ekf_on = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    def detect_fruit(self):
        if self.detector is None:
            warning = "No valid Checkpoint"
            cv2.putText(self.network_vis, warning, (40, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), thickness=2)
        elif self.command['inference'] and self.detector is not None:
            self.detector_output, self.network_vis = self.detector.detect_single_image(self.img)
            self.command['inference'] = False
            self.file_output = (self.detector_output, self.ekf)
            self.notification = f'{len(np.unique(self.detector_output))-1} fruit type(s) detected'
            
    def init_ekf(self, datadir, ip):
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
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
        if self.command['save_inference']:
            if self.file_output is not None:
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                        self.file_output[1])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False
            
    def draw(self, canvas):
        canvas.fill((0, 0, 0))
        text_colour = (200, 200, 200)
        pad = 40
        if self.ekf_on:
            ekf_view = self.ekf.draw_slam_state(res=(320, 480+pad))
        else:
            ekf_view = self.ekf.draw_slam_state(res=(320, 480+pad))/2
        self.draw_pygame_window(canvas, ekf_view, 
                                position=(2*pad+320, pad)
                                )
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, 
                                position=(pad, pad)
                                )
        detector_view = cv2.resize(self.network_vis,
                                   (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, 
                                position=(pad, 240+2*pad)
                                )
        canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2*pad+320, pad))
        self.put_caption(canvas, caption='Fruit Detector',
                         position=(pad, 240+2*pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(pad, pad))

        notifiation = TEXT_FONT.render(self.notification,
                                          False, text_colour)
        canvas.blit(notifiation, (60, 645))
        self.put_caption(canvas, caption='Notifiation:', position=(45, 600))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))
        
        
    def update_keyboard(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'][0] = min(self.command['motion'][0]+1, 1)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'][0] = max(self.command['motion'][0]-1, -1)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'][1] = min(self.command['motion'][1]+1, 1)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'][1] = max(self.command['motion'][1]-1, -1)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'WARNINIG!!! Press R again confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_markers = len(self.ekf.taglist)
                if n_markers == 0:
                    self.ekf_on = True
                    self.notification = 'SLAM is running'
                elif n_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()
        
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
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    
    width, height = 760, 760
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('RVSS 2021 Workshop')
    icon = pygame.image.load('pics/logo.png')
    pygame.display.set_icon(icon)
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/rvss_splash_8_bit.png')
    logo = pygame.image.load('pics/logo_8_bit.png')
    pygame.display.update()

    start = False

    counter = -20
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 660)
        canvas.blit(logo, (x_, 660))
        pygame.display.update()
        counter += 1

    operate = Operate(args)

    while start:
        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.detect_fruit()
        # visualise
        operate.draw(canvas)
        pygame.display.update()




