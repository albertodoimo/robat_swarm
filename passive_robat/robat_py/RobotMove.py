from thymiodirect import Connection
from thymiodirect import Thymio
from shared_queues import angle_queue, level_queue

import time
import random

class RobotMove():
    def __init__(self, forward_speed, turn_speed, left_sensor_threshold, right_sensor_threshold, critical_level, trigger_level, ground_sensors_bool = True):
        self.forward_speed = forward_speed
        self.turn_speed = turn_speed
        self.left_sensor_threshold = left_sensor_threshold
        self.right_sensor_threshold = right_sensor_threshold
        self.ground_sensors_bool = ground_sensors_bool
        self.critical_level = critical_level
        self.trigger_level = trigger_level
        self.running = True

        print("Initializing Thymio Robot")
        port = Connection.serial_default_port()
        th = Thymio(serial_port=port,
                    on_connect=lambda node_id: print(f'Thymio {node_id} is connected'))
        # Connect to Robot
        th.connect()
        self.robot = th[th.first_node()]

        # Delay to allow robot initialization of all variables
        # time.sleep(1)
        # b) print all variables
        # print(th.variables(th.first_node()))
        
        print("Robot connected")
        if self.ground_sensors_bool:
            print('ground.delta  L R = ', self.robot['prox.ground.delta'])
            print('ground.reflected  L R = ', self.robot['prox.ground.reflected'])

        self.stop_bool = False

    def angle_to_time(self, angle, forward_speed):
        #calculate the time needed to turn the robot by a certain angle
        A = 612.33
        B = -0.94
        t = A*forward_speed**B    
        return t * abs(angle) / 360 # time to turn by angle in seconds


    def audio_move(self):
        while self.running:
            try:
                # Check for a global stop signal (e.g., if the robot is lifted)
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    self.stop()
                    continue  # Go back to top of loop

                self.avoid_white_line()

                # Flush the angle queue to obtain the latest angle value.
                #  angle = None
                while not angle_queue.empty():
                    angle = angle_queue.get()
                    #print('Angle move:', angle)
                if angle is None:
                    #print('Empty queue: no angle value')
                    self.move_forward()  # Go straight if no angle is available.
                    continue

                # Flush the level queue similarly to get the latest level value.
                # level = None
                while not level_queue.empty():
                    level = level_queue.get()

                # Make a decision based on the latest values.

                if level is not None and level < self.critical_level and level > self.trigger_level:
                    self.robot["leds.top"] = [0, 0, 255]
                    self.robot["leds.bottom.right"] = [0, 0, 255]
                    self.robot["leds.bottom.left"] = [0, 0, 255]
                    print('2.1: angle=', angle)
                    if angle < 0:
                        print('3: Negative angle received, rotating left')
                        self.rotate_left(angle)
                    else:
                        print('4: Positive or zero angle received, rotating right')
                        self.rotate_right(angle)
                elif level is not None and level > self.critical_level:
                    self.robot["leds.top"] = [255, 0, 0]
                    self.robot["leds.bottom.right"] = [255, 0, 0]
                    self.robot["leds.bottom.left"] = [255, 0, 0]
                    if angle < 0:
                        print('5: Negative angle received, rotating right')
                        self.stop()
                        self.robot["leds.bottom.right"] = [255, 0, 0]
                        self.robot["leds.bottom.left"] = [255, 0, 0]  
                        self.robot["leds.top"] = [255, 0, 0]                  
                        
                        self.rotate_right(angle)
                        time.sleep(0.5)
                        
                    else:
                        print('6: Positive angle received, rotating left')
                        self.stop()
                        self.robot["leds.bottom.right"] = [255, 0, 0]
                        self.robot["leds.bottom.left"] = [255, 0, 0]
                        self.robot["leds.top"] = [255, 0, 0]
                                                
                        self.rotate_left(angle)
                        time.sleep(0.5)

                else:
                    pass

                # After executing a turn, go back to moving straight.
                #print("Returning to forward movement")
                self.move_forward()
                level = None

            except Exception as e:
                print('Error in audio_move:', e)
                self.stop_bool = True
            except KeyboardInterrupt:
                self.stop()
        else:
            self.stop()

    def move_forward(self):
        self.robot["leds.top"] = [0, 255, 0]
        self.robot["leds.bottom.right"] = [0, 255, 0]
        self.robot["leds.bottom.left"] = [0, 255, 0]        
        if self.robot is not None:
            if self.check_stop_all_motion():
                #print("stop all motion: move forward")
                self.stop_bool = True
                self.stop()
                # interrupt the loop
                return
            self.robot['motor.left.target'] = self.forward_speed
            self.robot['motor.right.target'] = self.forward_speed
            #if self.obstacle_ahead():
            #    self.collision_avoidance()
            #    continue
        else:
            print("self.robot is None")
            self.stop_bool = True
            self.stop()

    def rotate_right(self, angle):
        if self.robot is not None:
            counter = self.angle_to_time(angle, self.forward_speed)*1000
            while counter > 0:
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    break
                #print("rotate right with angle:", angle)
                #print('turn speed', abs(int(1/90 * (self.forward_speed * int(angle)))))
                #self.robot['motor.left.target'] = abs(int(1/90 * (self.forward_speed * int(angle))))
                #self.robot['motor.right.target'] = -abs(int(1/90 * (self.forward_speed * int(angle))))
                self.robot['motor.left.target']= self.forward_speed
                self.robot['motor.right.target']= -self.forward_speed
                
                counter -= 1
            else:
                self.robot['motor.left.target'] = 0
                self.robot['motor.right.target'] = 0
        else:
            self.stop()

    def rotate_left(self, angle):
        if self.robot is not None:
            counter = self.angle_to_time(angle, self.forward_speed)*1000
            while counter > 0:
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    break
                #print("rotate left with angle:", angle)
                #print('turn speed', abs(int(1/90 * (self.forward_speed * int(angle)))))
                #self.robot['motor.left.target'] = -abs(int(1/90 * (self.forward_speed * int(angle))))
                #self.robot['motor.right.target'] = abs(int(1/90 * (self.forward_speed * int(angle))))
                self.robot['motor.left.target']= -self.forward_speed
                self.robot['motor.right.target']= self.forward_speed
                counter -= 1
            else:
                self.robot['motor.left.target'] = 0
                self.robot['motor.right.target'] = 0
        else:
            self.stop()


    def move_back(self): # move back when robot sees a obstacle in front
        if self.robot is not None:
            counter = 4000
            while counter > 0:
                if self.check_stop_all_motion():
                    print("stop all motion: move back")
                    self.stop_bool = True
                    break
                if self.robot['prox.horizontal'][5] > 600 or self.robot['prox.horizontal'][6] > 600:
                    self.stop_bool = True
                    self.stop()
                self.robot['motor.left.target'] = -self.forward_speed
                self.robot['motor.right.target'] = -self.forward_speed
                counter -= 1
            else:
                # print("robot stop")
                self.robot['motor.left.target'] = 0
                self.robot['motor.right.target'] = 0
        else:
            self.stop()

    def check_stop_all_motion(self):
        if self.robot is not None:
            #print(f"Prox 0: {self.robot['prox.ground.delta'][0]}, Prox 1: {self.robot['prox.ground.delta'][1]}")
            if self.robot['prox.ground.delta'][0] < 1 or self.robot['prox.ground.delta'][1] < 10:
                #print("Robot lifted")
                return True
        else:
            self.stop()
            return False

    def avoid_white_line(self):
        if self.robot is not None:
            # check if the white line is detected
            if self.robot['prox.ground.reflected'][0] > self.left_sensor_threshold  and self.robot['prox.ground.reflected'][1] > self.right_sensor_threshold:
                # Both sensors detect the line
                #print('line detected L and R')
                counter = self.angle_to_time(90, self.forward_speed)
                while counter > 0:
                    if self.check_stop_all_motion():
                        self.stop_bool = True
                        break
                    self.robot['motor.left.target'] = self.turn_speed
                    self.robot['motor.right.target'] = -self.turn_speed
                    counter -= 1
                return
            if self.robot['prox.ground.reflected'][0] > self.left_sensor_threshold:
                # Left sensor detects the line
                #print('line detected L')
                counter = self.angle_to_time(90, self.forward_speed)
                while counter > 0:
                    if self.check_stop_all_motion():
                        #print("stop all motion: rotate right")
                        self.stop_bool = True
                        break
                    self.robot['motor.left.target'] = self.turn_speed
                    self.robot['motor.right.target'] = -self.turn_speed
                    counter -= 1
            if self.robot['prox.ground.reflected'][1] > self.right_sensor_threshold:
                # Right sensor detects the line   
                #print('line detected R')
                counter = self.angle_to_time(90, self.forward_speed)
                while counter > 0:
                    if self.check_stop_all_motion():
                        #print("stop all motion: rotate right")
                        self.stop_bool = True
                        break
                    self.robot['motor.left.target'] = self.turn_speed
                    self.robot['motor.right.target'] = -self.turn_speed
                    counter -= 1
        else:
            self.stop()

    def stop(self):
        self.robot['motor.left.target'] = 0
        self.robot['motor.right.target'] = 0
        self.robot["leds.top"] = [0, 0, 0]
        self.robot["leds.bottom.right"] = [0, 0, 0]
        self.robot["leds.bottom.left"] = [0, 0, 0]

#     def collision_avoidance(self):
#         if self.robot is not None:
#             print("Starting Collision avoidance")
#             self.robot['leds.top'] = [32, 0, 0]
#             counter = 3000
#             print("Move back")
#             while counter > 0:
#                 if self.robot['prox.horizontal'][5] > 600 or self.robot['prox.horizontal'][6] > 600:
#                     self.stop()
#                     continue
#                 self.robot['motor.left.target'] = -200
#                 self.robot['motor.right.target'] = -200
#                 counter -= 1
#             counter = 2000
#             print("Turn")
#             while counter > 0:
#                 self.robot['motor.left.target'] = -300
#                 self.robot['motor.right.target'] = 300
#                 counter -= 1
#             self.robot['leds.top'] = [0, 32, 0]

#     def obstacle_ahead(self):
#         if self.robot is not None:
#             if (self.robot['prox.horizontal'][0] > 1000 or self.robot['prox.horizontal'][1] > 1000 or
#                     self.robot['prox.horizontal'][2] > 1000 or self.robot['prox.horizontal'][3] > 1000 or self.robot['prox.horizontal'][4] > 1000):
#                 return True
#         return False

    def random_move(self):
        while True:
            if self.stop_bool:
                print("Stopping Robot")
                self.stop()
                break
            self.move_forward()
            if bool(random.getrandbits(1)):  
                self.rotate_right(angle)
            else:
                self.rotate_left(angle)

    def random_turn(self):
        while True:
            if self.check_stop_all_motion():
                self.stop_bool = True
                break
            if self.stop_bool:
                print("Stopping Robot")
                self.stop()
                break
            if bool(random.getrandbits(1)):  
                print('random turn right')
                self.rotate_right(angle)
                break
            else:
                print('random turn left')
                self.rotate_left()
                break
