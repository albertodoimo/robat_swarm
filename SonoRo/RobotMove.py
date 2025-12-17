#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created: 2025-12-17
Author: Alberto Doimo
email: alberto.doimo@uni-konstanz.de

"""
from thymiodirect import Connection
from thymiodirect import Thymio
from shared_queues import angle_queue, level_queue

import time
import random


class RobotMove:
    """This class manages robot motor control, LED indicators, ground sensors,
    and implements various behavioral modes

    Parameters
    ----------
    forward_speed : int
        Target speed for forward motor movement.
    turn_speed : int
        Target speed for rotational motor movement.
    left_sensor_threshold : int
        Threshold value for left ground sensor to detect white lines.
    right_sensor_threshold : int
        Threshold value for right ground sensor to detect white lines.
    trigger_level : float
        Sound pressure level (dB SPL) threshold to trigger detection.
    critical_level : float
        Critical sound pressure level (dB SPL) for avoiding collisions.
    ground_sensors_bool : bool, optional
        If True, initializes and monitors ground sensor values at start.

    Attributes
    ----------
    running : bool
        Flag indicating if the robot's main loop is running.
    turning_angle : int
        Default fixed turning angle in degrees (10 degrees).
    stop_bool : bool
        Flag indicating if robot should stop motion.
    robot : object
        Thymio robot instance for motor and sensor control.

    """

    def __init__(
        self,
        forward_speed,
        turn_speed,
        left_sensor_threshold,
        right_sensor_threshold,
        critical_level,
        trigger_level,
        ground_sensors_bool=True,
    ):
        self.forward_speed = forward_speed
        self.turn_speed = turn_speed
        self.left_sensor_threshold = left_sensor_threshold
        self.right_sensor_threshold = right_sensor_threshold
        self.ground_sensors_bool = ground_sensors_bool
        self.critical_level = critical_level
        self.trigger_level = trigger_level

        self.running = True
        self.turning_angle = 10  # degrees
        self.stop_bool = False

        print("Initializing Thymio Robot")
        port = Connection.serial_default_port()
        th = Thymio(
            serial_port=port,
            on_connect=lambda node_id: print(f"Thymio {node_id} is connected"),
        )
        # Connect to Robot
        th.connect()
        self.robot = th[th.first_node()]

        # Print all variables
        # print(th.variables(th.first_node()))
        # time.sleep(1)

        print("Robot connected")

        if self.ground_sensors_bool:
            print("\nGround sensors values...\n")
            time.sleep(2)  # wait for sensors to initialize
            print("ground.delta  L R = ", self.robot["prox.ground.delta"])
            print("ground.reflected  L R = ", self.robot["prox.ground.reflected"])
            print("\n")

    def angle_to_time(self, angle, speed):
        """Calculate the time needed to turn the robot by a specified angle.

        Parameters
        ----------
        angle : float
            The rotation angle in degrees.
        speed : float
            The rotational speed parameter.

        Returns
        -------
        float
            The time required to turn by the specified angle, in seconds.

        Notes
        -----
        A = 612.33 and B = -0.94 are empirically determined constants.

        """
        # calculate the time needed to turn the robot by a certain angle
        A = 612.33
        B = -0.94
        t = A * speed**B

        return t * abs(angle) / 360  # time to turn by angle in seconds

    def attraction_only(self):
        """Attraction-based movement control loop for the robot.

        Fixed angle attraction by self.turning_angle degrees turning.
        No reaction to critical level.

        Attributes Used
        ----------------
        self.running : bool
            Flag indicating if the robot's main loop is running.
        trigger_level : float
            Sound pressure level (dB SPL) threshold to trigger detection.
        critical_level : float
            Critical sound pressure level (dB SPL) for avoiding collisions.
        angle_queue : queue.Queue
            Inter-thread queue for sharing detected angles from AudioProcessor.
        level_queue : queue.Queue
            Inter-thread queue for sharing SPL levels from AudioProcessor.
        self.robot : object
            Thymio robot instance for motor and sensor control.

        Notes
        -----
        - The robot's  yellow LEDs indicate DOA direction in 22 degrees steps
        - When no angle data is available, Top and bottom LEDs illuminate green and the robot moves forward
        - Top and bottom LEDs turn blue when sound is above trigger level and red at critical level

        """

        while self.running:
            try:
                # Check for a global stop signal (e.g., if the robot is lifted)
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    self.stop()
                    continue  # Go back to top of loop

                self.avoid_white_line()

                while not angle_queue.empty():
                    angle = angle_queue.get()
                    if angle is not None:
                        if angle > -22 and angle < 22:
                            self.robot["leds.circle"] = [255, 0, 0, 0, 255, 0, 0, 0]
                        elif angle > 22 and angle < 67:
                            self.robot["leds.circle"] = [0, 255, 0, 255, 0, 0, 0, 0]
                        elif angle > 67:
                            self.robot["leds.circle"] = [0, 0, 255, 0, 0, 0, 0, 0]
                        elif angle < -22 and angle > -67:
                            self.robot["leds.circle"] = [0, 0, 0, 0, 0, 255, 0, 255]
                        elif angle < -67:
                            self.robot["leds.circle"] = [0, 0, 0, 0, 0, 0, 255, 0]

                    elif angle is None:
                        self.robot["leds.circle"] = [
                            255,
                            255,
                            255,
                            255,
                            255,
                            255,
                            255,
                            255,
                        ]
                        self.move_forward()  # Go straight if no angle is available.
                        continue

                # Flush the level queue similarly to get the latest level value.
                while not level_queue.empty():
                    level = level_queue.get()

                # Make a decision based on the latest values.
                if level is not None and level > self.trigger_level:
                    self.robot["leds.top"] = [0, 0, 255]
                    self.robot["leds.bottom.right"] = [0, 0, 255]
                    self.robot["leds.bottom.left"] = [0, 0, 255]
                    # print('2.1: angle=', angle)
                    if angle < 0:
                        self.smooth_rotate_left(self.turning_angle)
                        # Wait for the rotation to complete before continuing the loop
                        continue
                    else:
                        self.smooth_rotate_right(self.turning_angle)
                        continue
                elif level is not None and level > self.critical_level:
                    self.robot["leds.top"] = [255, 0, 0]
                    self.robot["leds.bottom.right"] = [255, 0, 0]
                    self.robot["leds.bottom.left"] = [255, 0, 0]
                    self.stop()
                    break
                else:
                    pass

                # After executing a turn, go back to moving straight.
                self.move_forward()
                level = None

            except Exception as e:
                self.stop_bool = True
                self.stop()
            except KeyboardInterrupt:
                self.stop_bool = True
                self.stop()
        else:
            self.stop_bool = True
            self.stop()

    def repulsion_only(self):
        """Execute repulsion-based movement behaviour for the robot.

        Fixed angle repulsion by self.turning_angle degrees turning.

        Attributes Used
        ----------------
        self.running : bool
            Flag indicating if the robot's main loop is running.
        critical_level : float
            Critical sound pressure level (dB SPL) for avoiding collisions.
        angle_queue : queue.Queue
            Inter-thread queue for sharing detected angles from AudioProcessor.
        level_queue : queue.Queue
            Inter-thread queue for sharing SPL levels from AudioProcessor.
        self.robot : object
            Thymio robot instance for motor and sensor control.
        self.turning_angle : int
            Fixed turning angle in degrees.

        Notes
        -----
        - The robot's circular yellow LEDs indicate DOA direction in 22 degrees steps
        - When no angle data is available, Top and bottom LEDs illuminate green and the robot moves forward
        - Top and bottom LEDs turn blue when sound is above trigger level and red at critical level

        """
        while self.running:
            try:
                # Check for a global stop signal (e.g., if the robot is lifted)
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    self.stop()
                    continue  # Go back to top of loop

                self.avoid_white_line()

                while not angle_queue.empty():
                    angle = angle_queue.get()
                    if angle is not None:
                        if angle > -22 and angle < 22:
                            self.robot["leds.circle"] = [255, 0, 0, 0, 255, 0, 0, 0]
                        elif angle > 22 and angle < 67:
                            self.robot["leds.circle"] = [0, 255, 0, 255, 0, 0, 0, 0]
                        elif angle > 67:
                            self.robot["leds.circle"] = [0, 0, 255, 0, 0, 0, 0, 0]
                        elif angle < -22 and angle > -67:
                            self.robot["leds.circle"] = [0, 0, 0, 0, 0, 255, 0, 255]
                        elif angle < -67:
                            self.robot["leds.circle"] = [0, 0, 0, 0, 0, 0, 255, 0]

                    elif angle is None:
                        self.robot["leds.circle"] = [
                            255,
                            255,
                            255,
                            255,
                            255,
                            255,
                            255,
                            255,
                        ]
                        self.move_forward()  # Go straight if no angle is available.
                        continue

                # Flush the level queue similarly to get the latest level value.
                while not level_queue.empty():
                    level = level_queue.get()

                # Make a decision based on the latest values.
                if level is not None and level > self.critical_level:
                    self.robot["leds.top"] = [255, 0, 0]
                    self.robot["leds.bottom.right"] = [255, 0, 0]
                    self.robot["leds.bottom.left"] = [255, 0, 0]
                    if angle < 0:
                        self.robot["leds.bottom.right"] = [255, 0, 0]
                        self.robot["leds.bottom.left"] = [255, 0, 0]
                        self.robot["leds.top"] = [255, 0, 0]
                        self.smooth_rotate_right(self.turning_angle)

                    else:
                        self.robot["leds.bottom.right"] = [255, 0, 0]
                        self.robot["leds.bottom.left"] = [255, 0, 0]
                        self.robot["leds.top"] = [255, 0, 0]
                        self.smooth_rotate_left(self.turning_angle)
                else:
                    pass

                # After executing a turn, go back to moving straight.
                self.move_forward()
                level = None

            except Exception as e:
                self.stop_bool = True
                self.stop()
            except KeyboardInterrupt:
                self.stop_bool = True
                self.stop()
        else:
            self.stop_bool = True
            self.stop()

    def audio_move(self):
        """Execute attraction and repulsion-based movement behaviour for the robot.

        Attributes Used
        ----------------
        self.running : bool
            Flag indicating if the robot's main loop is running.
        critical_level : float
            Critical sound pressure level (dB SPL) for avoiding collisions.
        trigger_level : float
            Sound pressure level (dB SPL) threshold to trigger detection.
        angle_queue : queue.Queue
            Inter-thread queue for sharing detected angles from AudioProcessor.
        level_queue : queue.Queue
            Inter-thread queue for sharing SPL levels from AudioProcessor.
        self.robot : object
            Thymio robot instance for motor and sensor control.
        self.turning_angle : int
            Fixed turning angle in degrees.

        Notes
        -----
        - The robot's circular yellow LEDs indicate DOA direction in 22 degrees steps
        - When no angle data is available, Top and bottom LEDs illuminate green and the robot moves forward
        - Top and bottom LEDs turn blue when sound is above trigger level and red at critical level

        """
        while self.running:
            try:
                # Check for a global stop signal (e.g., if the robot is lifted)
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    self.stop()
                    continue  # Go back to top of loop

                self.avoid_white_line()

                # Flush the angle queue to obtain the latest angle value.
                while not angle_queue.empty():
                    angle = angle_queue.get()
                    if angle is not None:
                        if angle > -22 and angle < 22:
                            self.robot["leds.circle"] = [255, 0, 0, 0, 255, 0, 0, 0]
                        elif angle > 22 and angle < 67:
                            self.robot["leds.circle"] = [0, 255, 0, 255, 0, 0, 0, 0]
                        elif angle > 67:
                            self.robot["leds.circle"] = [0, 0, 255, 0, 0, 0, 0, 0]
                        elif angle < -22 and angle > -67:
                            self.robot["leds.circle"] = [0, 0, 0, 0, 0, 255, 0, 255]
                        elif angle < -67:
                            self.robot["leds.circle"] = [0, 0, 0, 0, 0, 0, 255, 0]

                    elif angle is None:
                        self.robot["leds.circle"] = [
                            255,
                            255,
                            255,
                            255,
                            255,
                            255,
                            255,
                            255,
                        ]
                        self.move_forward()  # Go straight if no angle is available.
                        continue

                # Flush the level queue similarly to get the latest level value.
                while not level_queue.empty():
                    level = level_queue.get()

                # Make a decision based on the latest values.
                if (
                    level is not None
                    and level < self.critical_level
                    and level > self.trigger_level
                ):
                    self.robot["leds.top"] = [0, 0, 255]
                    self.robot["leds.bottom.right"] = [0, 0, 255]
                    self.robot["leds.bottom.left"] = [0, 0, 255]
                    if angle < 0:
                        self.rotate_left(self.turning_angle)
                        # Wait for the rotation to complete before continuing the loop
                        continue
                    else:
                        self.rotate_right(self.turning_angle)
                        continue
                elif level is not None and level > self.critical_level:
                    self.robot["leds.top"] = [255, 0, 0]
                    self.robot["leds.bottom.right"] = [255, 0, 0]
                    self.robot["leds.bottom.left"] = [255, 0, 0]
                    if angle < 0:
                        self.robot["leds.bottom.right"] = [255, 0, 0]
                        self.robot["leds.bottom.left"] = [255, 0, 0]
                        self.robot["leds.top"] = [255, 0, 0]
                        self.rotate_right(self.turning_angle)
                    else:
                        self.robot["leds.bottom.right"] = [255, 0, 0]
                        self.robot["leds.bottom.left"] = [255, 0, 0]
                        self.robot["leds.top"] = [255, 0, 0]
                        self.rotate_left(self.turning_angle)

                else:
                    pass
                # After executing a turn, go back to moving straight.
                self.move_forward()
                level = None

            except Exception as e:
                self.stop_bool = True
                self.stop()

            except KeyboardInterrupt:
                self.stop_bool = True
                self.stop()
        else:
            self.stop_bool = True
            self.stop()

    def move_forward(self):
        """Move the robot forward at the specified forward speed.

        Attributes Used
        ------
        self.forward_speed : int
            Target speed for forward motor movement.
        self.robot : object
            Thymio robot instance for motor and sensor control.

        Notes
        -----
        -Top and bottom LEDs illuminate green and the robot moves forward

        """

        self.robot["leds.top"] = [0, 255, 0]
        self.robot["leds.bottom.right"] = [0, 255, 0]
        self.robot["leds.bottom.left"] = [0, 255, 0]
        if self.robot is not None:
            if self.check_stop_all_motion():
                self.stop_bool = True
                self.stop()
                # interrupt the loop
                return
            self.robot["motor.left.target"] = self.forward_speed
            self.robot["motor.right.target"] = self.forward_speed

        else:
            self.stop_bool = True
            self.stop()

    def rotate_right(self, angle):
        """Rotate the robot to the right by a specified angle.

        Parameters
        ----------
        angle : float
            The angle in degrees to rotate the robot.

        Attributes Used
        ----------------
        self.robot : object
            Thymio robot instance for motor and sensor control.
        self.turn_speed : int
            Target speed for rotational motor movement.

        Notes
        -----
        - Gives priority to the line detection to avoid the robot exiting the arena

        """
        if self.robot is not None:
            counter = self.angle_to_time(angle, self.turn_speed)
            start_time = time.time()
            while time.time() - start_time < counter:
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    break
                self.robot["motor.left.target"] = self.turn_speed
                self.robot["motor.right.target"] = -self.turn_speed
        else:
            self.stop()

    def rotate_left(self, angle):
        """Rotate the robot to the left by a specified angle.

        Parameters
        ----------
        angle : float
            The angle in degrees to rotate the robot.

        Attributes Used
        ----------------
        self.robot : object
            Thymio robot instance for motor and sensor control.
        self.turn_speed : int
            Target speed for rotational motor movement.

        Notes
        -----
        - Gives priority to the line detection to avoid the robot exiting the arena

        """
        if self.robot is not None:
            counter = self.angle_to_time(angle, self.turn_speed)
            start_time = time.time()
            while time.time() - start_time < counter:
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    break
                self.robot["motor.left.target"] = -self.turn_speed
                self.robot["motor.right.target"] = self.turn_speed

        else:
            self.stop()

    def smooth_rotate_right(self, angle):
        """Rotate the robot to the right by a specified angle but keeps some forward speed while turning.

        Parameters
        ----------
        angle : float
            The angle in degrees to rotate the robot to the right.

        Attributes Used
        ----------------
        self.robot : object
            Thymio robot instance for motor and sensor control.
        self.turn_speed : int
            Target speed for rotational motor movement.

        Notes
        -----
        - Gives priority to the line detection to avoid the robot exiting the arena

        """
        if self.robot is not None:
            counter = self.angle_to_time(angle, self.turn_speed)
            start_time = time.time()
            while time.time() - start_time < counter:
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    break
                self.robot["motor.left.target"] = self.turn_speed + self.turn_speed // 2
                self.robot["motor.right.target"] = (
                    -self.turn_speed + self.turn_speed // 2
                )
        else:
            self.stop()

    def smooth_rotate_left(self, angle):
        """Rotate the robot to the left by a specified angle but keeps some forward speed while turning.

        Parameters
        ----------
        angle : float
            The angle in degrees to rotate the robot to the right.

        Attributes Used
        ----------------
        self.robot : object
            Thymio robot instance for motor and sensor control.
        self.turn_speed : int
            Target speed for rotational motor movement.

        Notes
        -----
        - Gives priority to the line detection to avoid the robot exiting the arena

        """
        if self.robot is not None:
            counter = self.angle_to_time(angle, self.turn_speed)
            start_time = time.time()
            while time.time() - start_time < counter:
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    break
                self.robot["motor.left.target"] = (
                    -self.turn_speed + self.turn_speed // 2
                )
                self.robot["motor.right.target"] = (
                    self.turn_speed + self.turn_speed // 2
                )
        else:
            self.stop()

    def move_back(self):
        """Move the robot backward when an obstacle is detected in front.

        Attributes Used
        ---------------
        self.robot : object
            Thymio robot instance for motor and sensor control.

        """

        if self.robot is not None:
            counter = 0.2  # sec
            start_time = time.time()
            while time.time() - start_time < counter:
                self.robot["motor.left.target"] = -self.forward_speed
                self.robot["motor.right.target"] = -self.forward_speed
        else:
            self.stop_bool = True
            self.stop()

    def check_stop_all_motion(self):
        """Check if the robot should stop all motion based on ground proximity sensors.

        Attributes Used
        ---------------
        self.robot : object
            Thymio robot instance for motor and sensor control.

        Returns
        -------
        bool
            True if the robot is lifted (proximity delta < 10 on either sensor),
            False if robot object is None (after calling stop()).

        """
        if self.robot is not None:
            if (
                self.robot["prox.ground.delta"][0] < 10
                or self.robot["prox.ground.delta"][1] < 10
            ):
                return True
        else:
            self.stop_bool = True
            self.stop()
            return False

    def avoid_white_line(self):
        """Avoid white line detection and response.

        If both ground sensors detect a white line, the robot performs a random turn.
        If only the left sensor detects a white line, the robot turns right.
        If only the right sensor detects a white line, the robot turns left.

        Attributes Used
        ---------------
        self.left_sensor_threshold : int
            Threshold value for left ground sensor to detect white lines.
        self.right_sensor_threshold : int
            Threshold value for right ground sensor to detect white lines.
        self.robot : object
            Thymio robot instance for motor and sensor control.
        self.stop_bool : bool
            Set to True when stop condition is met

        """

        if self.robot is not None:
            # check if the white line is detected
            if (
                self.robot["prox.ground.reflected"][0] > self.left_sensor_threshold
                and self.robot["prox.ground.reflected"][1] > self.right_sensor_threshold
            ):
                # Both sensors detect the line
                if self.check_stop_all_motion():
                    self.stop_bool = True
                    self.stop()
                    return
                self.random_turn(90)

            elif self.robot["prox.ground.reflected"][0] > self.left_sensor_threshold:
                # Left sensor detects the line
                counter = self.angle_to_time(50, self.turn_speed)
                start_time = time.time()
                while time.time() - start_time < counter:
                    if self.check_stop_all_motion():
                        self.stop_bool = True
                        break
                    self.robot["motor.left.target"] = self.turn_speed
                    self.robot["motor.right.target"] = -self.turn_speed

            elif self.robot["prox.ground.reflected"][1] > self.right_sensor_threshold:
                # Right sensor detects the line
                counter = self.angle_to_time(50, self.turn_speed)
                start_time = time.time()
                while time.time() - start_time < counter:
                    if self.check_stop_all_motion():
                        self.stop_bool = True
                        break
                    self.robot["motor.left.target"] = -self.turn_speed
                    self.robot["motor.right.target"] = self.turn_speed

        else:
            self.stop()

    def stop(self):
        """Stop all robot movement and turn off all LEDs.

        This method stops the robot by setting both motor targets to zero and
        disables all LED indicators by setting them to off (black color).

        """
        self.robot["motor.left.target"] = 0
        self.robot["motor.right.target"] = 0
        self.robot["leds.top"] = [0, 0, 0]
        self.robot["leds.bottom.right"] = [0, 0, 0]
        self.robot["leds.bottom.left"] = [0, 0, 0]
        self.robot["leds.circle"] = [0, 0, 0, 0, 0, 0, 0, 0]

    def random_turn(self, angle):
        """Perform a random movement of the robot in a random direction.

        Parameters
        ----------
        angle : float
            The rotation angle in degrees

        Attributes Used
        ---------------
        self.robot : object
            Thymio robot instance for motor and sensor control.
        self.stop_bool : bool
            Set to True when stop condition is met

        """
        if self.robot is not None:
            if self.check_stop_all_motion():
                self.stop_bool = True
                self.stop()
                return
            if bool(random.getrandbits(1)):
                self.rotate_right(angle)
                return
            else:
                self.rotate_left(angle)
                return
