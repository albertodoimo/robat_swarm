from thymiodirect import Connection
from thymiodirect import Thymio

port = Connection.serial_default_port()
th = Thymio(serial_port=port,
                on_connect=lambda node_id: print(f'Thymio {node_id} is connected'))
# Connect to Robot
th.connect()
robot = th[th.first_node()]
robot['motor.left.target'] = 0
robot['motor.right.target'] = 0
print("Robot stopped")