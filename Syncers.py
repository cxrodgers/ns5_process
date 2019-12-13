"""Module containing objects to sync data"""
from __future__ import absolute_import
from __future__ import division
from past.utils import old_div
import numpy as np
import scipy.signal
from . import AudioTools


def find_audio_onsets_by_bout(rs=None, raudio=None, stop_at=None, 
    inter_tone=6000, tones_per_bout=39, dsratio=1, bout_refractory=290000,
    threshold=None):
    """Pipeline for getting audio onsets from a bout structure.
    
    This assumes that the audio trace consists of bouts occuring at variable
    latencies, each of which consists of a fixed number of tones,
    separated by a fixed interval.
    
    The first tone in each bout is detected by threshold crossing. Subsequent
    tone onsets within that bout are calculated using the parameters. Any
    other threshold crossings within the bout are ignored.
    
    This works well for tonetask data, where the first tone can reliably
    be detected, but not every tone. (This is because some low-frequency
    drift / decay occurs between bouts, and because tones are variable power,
    so that the weakest tone is below the highest drift.)
    
    This gives more reliable results if highpassing the data first; however
    this is not an option if the tones may be aliased into the rejection
    bandwidth of the filter.
    
    Arguments
    ---------
    rs : RecordingSession from which to get data
    raudio : if rs is None, use this 2xN array as the audio data
    stop_at : truncate audio data at this many samples
    inter_tone : fixed distance between tones in a bout
    tones_per_bout : number of tones in each bout
    dsratio : integer to downsample by before detecting tones from the filtered
        signal. Actually not so useful since the smoothing is the time-limiting
        step.
    bout_refractory : minimum time after bout onset before another bout
        can be detected
    threshold : value of the smoothed signal to detect tones at, in dB
        if None, use the median of 10**log10(smoothed_signal)
    
    Returns
    -------
    starts : Detected/inferred tone onsets in samples
    dsmoothed : Smoothed signal on which detection was done (for debugging)
    threshold : threshold used (for debugging)    
    
    Here is a good way to check the performance while debugging. Works
    best if dsratio >> 1 and/or stop_at << 1e6.
    plt.plot(dsmoothed)
    plt.plot(starts, threshold*np.ones_like(starts), 'r*')
    plt.show()
    And view in a log basis.
    """
    # Grab raw data
    if rs is not None:
        # RS provided
        l = rs.get_ns5_loader()
        l.load_file()
        
        if stop_at is None:
            stop_at = l.header.n_samples
        
        res = AudioTools.check_audio_alignment(rs, timestamps=[0], 
            dstart=0, dstop=stop_at, also_smooth=False, plot=False)
        raudio = res['raw'].squeeze()
    else:
        # No RS provided
        if stop_at is not None:
            raudio = raudio[:, :stop_at]
    
    # Smooth and downsmaple
    smoothed = smooth(raudio)
    dsmoothed = smoothed[:, ::dsratio].mean(axis=0)
    
    # Set threshold
    if threshold is None:
        threshold = 10 ** np.median(np.log10(dsmoothed[dsmoothed > 0]))
    
    # Find bout starts
    bout_starts = find_threshold_crossings(
        dsmoothed, threshold, old_div(bout_refractory,dsratio))
    
    # Subslice to get tone starts
    starts = subslice(bout_starts, 0, within_bout_num=tones_per_bout, 
        within_bout_inter=(old_div(inter_tone, dsratio)))
    
    return starts, dsmoothed, threshold


# Tools
def smooth(signal, gstd=100, glen=None):
    if glen is None:
        glen = int(2.5 * gstd)
    gb = scipy.signal.gaussian(glen * 2 , gstd, sym=False)[glen:]
    gb = old_div(gb, np.sum(gb ** 2))
    signal2 = scipy.signal.filtfilt(gb, [1], signal ** 2)
    
    return signal2

def find_threshold_crossings(signal, threshold, refractory):
    # Threshold
    above_thresh = np.where(signal > threshold)[0]

    # Iterate through and grab tone starts
    starts_l = []
    while len(above_thresh) > 0:
        # Add first crossing
        starts_l.append(above_thresh[0])
        
        # Remove all crossings within rejection window
        above_thresh = above_thresh[
            above_thresh > (starts_l[-1] + refractory)]
    starts = np.asarray(starts_l)    
    
    return starts

def subslice(bout_starts, within_bout_start=0, within_bout_num=39,
    within_bout_inter=6000):
    """Sub slice bouts"""
    # Set up within bout
    within_bout = np.arange(within_bout_num) * within_bout_inter
    within_bout += within_bout_start
    return np.concatenate(
        [bout_start + within_bout for bout_start in bout_starts])


# Not working


def highpass_smooth_threshold(raudio, dsratio=100):
    """Highpass data before smoothing and thresholding.
    
    Doesn't work for targets near Nyquist
    """
    # Remove drift
    b, a = scipy.signal.butter(2, old_div(500,15e3), btype='high')
    fraudio = scipy.signal.filtfilt(b, a, raudio)
    
    smoothed = smooth(fraudio)
    smoothed = smoothed[::dsratio]
    
    return find_threshold_crossings(smoothed, threshold=10e3, refractory=old_div(6e3,dsratio))
