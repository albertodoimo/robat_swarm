
def get_card(device_list):
    """
    Get the index of the ASIO card in the device list.
    Parameters:
    - device_list: list of devices (usually = sd.query_devices())

    Returns: index of the card in the device list
    """
    for i, each in enumerate(device_list):
        dev_name = each['name']
        name = 'MCHStreamer' in dev_name
        if name:
            return i
    return None