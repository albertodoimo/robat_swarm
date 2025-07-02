# imports
import argparse
import time
import math
import random

# import thymiodirect package
from thymiodirect import Connection 
from thymiodirect import Thymio

def main(use_sim=False, ip='localhost', port=2001):
    ''' Main function '''

    try:
        # Configure Interface to Thymio robot
        # simulation
        if use_sim:
            th = Thymio(use_tcp=True, host=ip, tcp_port=port, 
                        on_connect=lambda node_id: print(f' Thymio {node_id} is connected'))
        # real robot
        else:
            port = Connection.serial_default_port()
            th = Thymio(serial_port=port, 
                        on_connect=lambda node_id: print(f'Thymio {node_id} is connected'))

        # Connect to Robot
        th.connect()
        robot = th[th.first_node()]

        # Delay to allow robot initialization of all variables
        time.sleep(1)
        
        # variables
        state = 'find'
        ts = time.time()
        direction = random.choice(['left', 'right'])

        # Main loop
        while True:
       
            ground_sensors = robot['prox.ground.reflected']
            ground_sensors_max = 1000
            #ground_sensors = ground_sensors_max - ground_sensors
            # Adjust these threshold values as needed
            ground_sensors = robot['prox.ground.reflected']
            #print('ground = ',robot['prox.ground.reflected'])
            # Adjust these threshold values as needed
            left_sensor_threshold = 250
            right_sensor_threshold = 250

            #TODO: 4.3 random walk
            current_time = time.time()
            detectsCollision = max([robot['prox.horizontal'][i] > 800 for i in range(5)])
            # print(robot['prox.horizontal'])

            if detectsCollision > 0: 
                robot['motor.left.target'] = -150
                robot['motor.right.target'] = -150
                #print('obstacle detected')
                state = 'turn'
                ts = current_time
            elif state == 'find':
                # Drive straight for 5 seconds
                robot['motor.left.target'] = 250
                robot['motor.right.target'] = 250
                if current_time - ts >= 5:
                    # Change state to turn
                    state = 'turn'
                    ts = current_time
                    direction = random.choice(['left', 'right'])
                elif ground_sensors[0] > left_sensor_threshold  and ground_sensors[1]> right_sensor_threshold:
                    # Both sensors detect the line, turn
                    robot['motor.left.target'] = 0
                    robot['motor.right.target'] = 0
                    state = 'turn'
                elif ground_sensors[0] < left_sensor_threshold and ground_sensors[1] > right_sensor_threshold:
                    # Only right sensor detects the line, turn left
                    robot['motor.left.target'] = -120
                    robot['motor.right.target'] = 100
                    state = 'find'
                elif ground_sensors[0] > left_sensor_threshold and ground_sensors[1] < right_sensor_threshold:
                    # Only left sensor detects the line, turn right
                    robot['motor.left.target'] = 100 
                    robot['motor.right.target'] = -120               
                    state = 'find'
            elif state == 'turn':
                # Turn in a random direction for 1 second
                if direction == 'left':
                    robot['motor.left.target'] = -150
                    robot['motor.right.target'] = 150
                else:
                    robot['motor.left.target'] = 150
                    robot['motor.right.target'] = -150
                if current_time - ts >= 1:
                    # Change state to find
                    state = 'find'
                    ts = current_time            
                                   
    except Exception as err:
        # Stop robot
        robot['motor.left.target'] = 0
        robot['motor.right.target'] = 0 
        print(err)
    except KeyboardInterrupt:
        robot['motor.left.target'] = 0
        robot['motor.right.target'] = 0
        print("Press Ctrl-C again to end the program")


if __name__ == '__main__':
    # Parse commandline arguments to cofigure the interface for a simulation (default = real robot)
    parser = argparse.ArgumentParser(description='Configure optional arguments to run the code with simulated Thymio. '
                                                    'If no arguments are given, the code will run with a real Thymio.')
    
    # Add optional arguments
    parser.add_argument('-s', '--sim', action='store_true', help='set this flag to use simulation')
    parser.add_argument('-i', '--ip', help='set the TCP host ip for simulation. default=localhost', default='localhost')
    parser.add_argument('-p', '--port', type=int, help='set the TCP port for simulation. default=2001', default=2001)

    # Parse arguments and pass them to main function
    args = parser.parse_args()
    main(args.sim, args.ip, args.port)