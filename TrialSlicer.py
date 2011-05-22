class TrialSlicer:
    """Intelligently determines trial starts and stops.
    
    Given a set of timestamps representing stimulus onsets, it will
    return a list of trial start times and trial stop times. 
    
    * Each stimulus onset gets its own trial.
    * No two trials are overlapping.
    * Time before and time after the onset is allocated as well as possible,
    according to provided parameters.
    """
    
    def __init__(self, timestamps, soft_limits, hard_limits):
        