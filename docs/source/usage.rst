Usage
=====

.. _installation:

Installation
------------

.. To use Lumache, first install it using pip:

.. .. code-block:: console

..    (.venv) $ pip install lumache

Logic Overview
----------------

By running  `SonoRo_swarm.py` on your SonoRo robots, AudioProcessor class starts streaming Direction of Arrival (DOA) of incoming audio to the robot and dB SPL values.

`SonoRo_swarm.py` then passes then to RobotMove class, which moves the robot's wheel accorsingly to perform the selected behaviour.

By selecting ``attraction`` behaviour, the robot will move towards the direction of the highest dB SPL value, while by selecting ``repulsion`` behaviour, the robot will move away from it.
The ``dynamic_movement`` parameter instead combined the two movements together using the selected dB SPL ``trigger_level`` and ``critical_level`` thresholds. When the dB SPL is above `critical` the threshold, the robot will move away from the sound source, while when it is above the `trigger` threshold, it will move towards it.

