# This file loads binary data and finds timestamps of audio events
import numpy as np
import ns5
import AudioTools
import os.path


def run(filename, manual_threshhold=None):
    """Given ns5 file, extracts audio onsets and writes to disk.
    
    Uses :class ns5.Loader to open ns5 file and :class 
    AudioTools.OnsetDetector to identify onsets.
    
    Input
    -----
    `filename` : path to *.ns5 binary file
    `manual_threshhold` : specify threshhold yourself
    
    Output
    ------
    Plaintext file TIMESTAMPS in same directory as input, containing
    detected stimulus onsets in samples
    """
    
    # Load ns5 file
    l = ns5.Loader(filename=filename)
    l.load_file()
    
    # Grab audio data, may be mono or stereo
    audio_data = l.get_all_analog_channels()

    # Instantiate an onset detector
    od = AudioTools.OnsetDetector(input_data=audio_data,
        F_SAMP=l.header.f_samp,
        plot_debugging_figures=False, 
        manual_threshhold=manual_threshhold,
        minimum_duration_ms=50)
    
    # Run it and save output to disk
    od.execute()
    
    do = od.detected_onsets
    np.savetxt(os.path.join(os.path.split(filename)[0], 'TIMESTAMPS'), do,
        fmt='%d')

    return do