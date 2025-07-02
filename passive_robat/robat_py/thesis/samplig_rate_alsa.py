import sounddevice as sd


S = sd.InputStream()
print('fs = ', S.samplerate)
print('blocksize = ', S.blocksize)
print('channels = ', S.channels)
print('latency = ', S.latency)
print('devinfo = ', S.device)
print(sd.query_devices())

import queue

def get_card(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i

usb_fireface_index = get_card(sd.query_devices())
print('usb_fireface_index=',usb_fireface_index)