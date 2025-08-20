import numpy as np

def avar_angle(delay_set,nchannels,mic_spacing,ref_channel):
    '''
    calculates the mean angle of arrival to the array with respect to reference channel
    Parameters:
    - delay_set: the time delay between signals
    - nchannels: number of mics in the array
    - mic_spacing: inter-distance between the mics
    - ref_channel: reference channel

    Returns:
    - avar_theta: the mean angle of arrival to the array 
    '''
    theta = []
    #print('delay_set=', delay_set)
    if ref_channel!=0: #centered reference that works with odd mics
        for each in range(0, nchannels//2):
            #print('\n1',each)
            #print('1',nchannels//2-each)
            theta.append(-np.arcsin((delay_set[each]*343)/((nchannels//2-each)*mic_spacing))) # rad
            i=nchannels//2-each
            #print('i=',i)
        for each in range(nchannels//2, nchannels-1):
            #print('\n2',each)
            #print('2',i)
            theta.append(np.arcsin((delay_set[each]*343)/((i)*mic_spacing))) # rad
            i+=1
    else:   
        for each in range(0, nchannels-1):
            theta.append(np.arcsin((delay_set[each]*343)/((each+1)*mic_spacing))) # rad
    #print('theta',theta)
    avar_theta = np.mean(theta)

    return avar_theta