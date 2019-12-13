from __future__ import absolute_import
# This file loads binary data and finds timestamps of audio events
import numpy as np
from . import ns5
from . import AudioTools
import os.path


def run(filename, manual_threshhold=None, audio_channels=None, 
    minimum_duration_ms=50, drop_first_and_last=False, debug_mode=False,
    truncate_data=None):
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
    if audio_channels is None:
        audio_data = l.get_all_analog_channels()
    else:
        # Specify channels manually
        audio_data = np.array(\
            [l.get_analog_channel_as_array(ch) for ch in audio_channels])
    
    # Plot the data, there may be artefacts to be truncated
    if debug_mode:
        import matplotlib.pyplot as plt
        plt.plot(audio_data.transpose())
        plt.show()
        1/0

    # Truncate artefacts
    if truncate_data is not None:
        audio_data = audio_data[:,:truncate_data]

    # Onset detector expect mono to be 1d, not Nx1
    if len(audio_channels) == 1:
        # Mono
        audio_data = audio_data.flatten()

    # Instantiate an onset detector
    od = AudioTools.OnsetDetector(input_data=audio_data,
        F_SAMP=l.header.f_samp,
        plot_debugging_figures=True, 
        manual_threshhold=manual_threshhold,
        minimum_duration_ms=minimum_duration_ms)
    
    # Run it and save output to disk
    od.execute()
    
    do = od.detected_onsets
    if drop_first_and_last:
        do = do[1:-1]
    np.savetxt(os.path.join(os.path.split(filename)[0], 'TIMESTAMPS'), do,
        fmt='%d')

    return do
