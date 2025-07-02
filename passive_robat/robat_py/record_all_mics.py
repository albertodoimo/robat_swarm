# file to record all the mics in the system in a single file

from functions.get_card import get_card 

#!/usr/bin/env python3
"""Create a recording with arbitrary duration.

The soundfile module (https://python-soundfile.readthedocs.io/)
has to be installed!

"""
import argparse
import tempfile
import queue
import sys
import datetime 
import os

import sounddevice as sd
import soundfile as sf
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)



def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)

parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'filename', nargs='?', metavar='FILENAME',
    help='audio file to store recording to')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-r', '--samplerate', type=int, help='sampling rate')
parser.add_argument(
    '-c', '--channels', type=int, default=1, help='number of input channels')
parser.add_argument(
    '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
parser.add_argument(
    '-a', '--angle', type=str, help='angle')
args = parser.parse_args(remaining)

q = queue.Queue()


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

# Create folder for saving recordings
timenow = datetime.datetime.now()
time = timenow.strftime('%Y-%m-%d')
save_path = '/home/thymio/robat_py/array_calibration/'
folder_name = str(time)
folder_path = os.path.join(save_path, folder_name)
os.makedirs(folder_path, exist_ok=True)

time1 = timenow.strftime('%Y-%m-%d__%H-%M-%S')
#name = 'MULTIWAV_' + str(time1) + '.wav'
name = args.angle + '.wav'
args.filename = os.path.join(folder_path, name)
args.device = get_card(sd.query_devices())
print(sd.query_devices())
print('device = ', args.device)
args.samplerate = 48000
args.channels = 5

try:
    if args.samplerate is None:  
        print('error!: no samplerate set! Using default')
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = int(device_info['default_samplerate'])
    if args.filename is None:
        timenow = datetime.datetime.now()
        time1 = timenow.strftime('%Y-%m-%d__%H-%M-%S')
        args.filename = 'MULTIWAV_' + str(time1) + '.wav'
    print('fs=',args.samplerate)

    # Make sure the file is opened before recording anything:
    with sf.SoundFile(args.filename, mode='x', samplerate=args.samplerate,
                      channels=args.channels, subtype=args.subtype) as file:
        with sd.InputStream(samplerate=args.samplerate, device=args.device,
                            channels=args.channels, callback=callback):
            print('#' * 80)
            print('press Ctrl+C to stop the recording')
            print('#' * 80)
            while True:
                file.write(q.get())
except KeyboardInterrupt:
    print('\nRecording finished: ' + repr(args.filename))
    parser.exit(0)
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))



