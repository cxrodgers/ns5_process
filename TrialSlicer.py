import numpy as np

def slice_trials(timestamps, soft_limits=(-100,100), hard_limits=(-1,1), 
    data_range=None, overlap=0, meth='end_of_previous'):
    """Intelligently determines trial starts and stops.
    
    Given a set of timestamps representing stimulus onsets, it will
    return a list of trial start times and trial stop times. 
    
    * Each stimulus onset gets its own trial.
    * No two trials are overlapping.
    * Time before and time after the onset is allocated as well as possible,
        according to provided parameters.
    
    Inputs
    ------
    timestamps : triggers, like stimulus onsets, one per trial
    soft_limits : a tuple (-pre, post) will capture:
        * (at most) `pre` samples before, and
        * (at most) `post` samples after each trigger, 
        * unless collisions occur.
    hard_limits : a tuple (-pre, post) will capture:
        * (at least) pre samples before, and
        * (at least) post samples after each trigger
        * and an error will occur if this is not possible.
        The default is (-1, 1) which means that an error will only occur
        if two triggers are consecutive.
    data_range : (min, max+1) of allowable indices to the underlying data.
        (0, len(my_data)) would be a good choice. Before any other processing,
        time stamps whose hard limits would place them outside of the data
        range are removed (with a warning).
    overlap : not yet implemented, but in some cases it might be nice to
        require or allow certain overlap between trials, for control of
        edge cases later.
    meth : string determining collision resolution method
        'end_of_previous' : Extra data will be allocated to previous
        'beginning_of_next' : Extra data will be allocated to next
        'split' : Extra data split 50/50 between adjacent segments
    
    Outputs
    -------
    Tuple (trial_starts, trial_stops) in samples. This will follow Python
    indexing convention such that the last index is not included. So if
    the first two trials are consecutive, then trial_stop[0] = trial_starts[1].
    """
    hard_limits = np.asarray(hard_limits, dtype=np.int)
    soft_limits = np.asarray(soft_limits, dtype=np.int)
    
    # Sort so we can assume consecutiveness later on
    timestamps.sort()
    
    # Deal with trials too close to the start or end to satisfy hard_limits
    # That is, if data_range is (0, 1000) and hard_limits is (-100, 200),
    # discard any time stamp <100 or >800
    if data_range is not None:
        data_range = np.asarray(data_range, dtype=np.int)
        too_small_mask = timestamps < (data_range[0] - hard_limits[0])
        too_large_mask = timestamps > (data_range[1] - hard_limits[1])        
        throw_away_mask = too_small_mask | too_large_mask
        if np.any(throw_away_mask):
            print ("warning: %d triggers are too close to boundaries" % \
                np.sum(throw_away_mask))
            timestamps = timestamps[np.logical_not(throw_away_mask)]
    
    # Check whether it is even possible to satisfy the constraints
    hard_starts, hard_stops = [timestamps + hl for hl in hard_limits]
    if np.any(hard_stops[:-1] > hard_starts[1:]):
        raise(ValueError(\
            "The hard_limits you provided are impossible to satisfy"))    

    # Begin greedily, by taking as much as each trigger wants
    final_starts, final_stops = [timestamps + sl for sl in soft_limits]

    # Deal seperately with first start and last stop, keeping them in data range
    if len(timestamps) > 0:
        final_starts[0] = max([final_starts[0], data_range[0]])
        final_stops[-1] = min([final_stops[-1], data_range[1]])
    
    # For the rest, collisions occur wherever stops[n] > starts[n+1]    
    # Maximum possible value in collisions is n_timestamps - 1
    # Note indexes are into final_stops. Add one to get index into starts.
    collisions, = np.nonzero(final_stops[:-1] > final_starts[1:])
    
    # Resolve collisions according to specified method
    if meth == 'end_of_previous':
        # Give the disputed territory to the end of the previous
        final_stops[collisions] = hard_starts[collisions + 1]
    elif meth == 'beginning_of_next':
        # Give the disputed territory to the start of the next
        final_stops[collisions] = hard_stops[collisions]   
    elif meth == 'split':
        # Split the disputed territory halfway (integer division)
        final_stops[collisions] = (hard_stops[collisions] + 
            hard_starts[collisions + 1]) / 2
    else:
        raise(ValueError("Not a valid collision resolution method"))
    
    # Assign the starts of the post-collision trials to be equal to the
    # stops of the pre-collision trials, thus ensuring no overlap.
    final_starts[collisions + 1] = final_stops[collisions]
    
    # Return calculated starts and stops
    return (final_starts, final_stops)
