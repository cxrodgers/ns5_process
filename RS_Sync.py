import numpy as np
from ns5_process import bcontrol
import kkpandas
import os.path
import collections

class RS_Syncer:
    """Defines preferred interface for accessing trial numbers and times
    
    This defines the "right" way to access the onset times and trial numbers
    
    Can return either neural or behavioral times
    
    Attributes
    btrial_numbers : Behavioral trial numbers that occurred during the
        recording of this session. This is defined by the index column of
        trials_info, which matches the displayed values in Matlab.
        The trial_%d_out states inserted into the events structure were named
        using this convention.
    
    ntrial_numbers : range(0, len(TIMESTAMPS)), that is, a 0-based counting
        of detected trials in neural data

    btrial_start_times : Start times of behavioral trial, length same as
        btrial_numbers, in behavioral time base
    
    Not implemented till I decide more:
    ntrial_start_times : Start times of OE Segments, length same as
        ntrial_numbers. Note not triggered on the same trial event as
        btrial_start_times. Not implemented yet till I decide whether it
        should actually refer to the digital trial pulse times
    
    btrial_onset_times : Onset of stimulus in behavioral trials
        Of length equal to btrial_numbers
    
    ntrial_onset_times : Onset of stimulus in neural trials
        Of length equal to ntrial_numbers. This one should actually be
        referencing the same physical event as btrial_onset_times
    
    Preferred accessor methods
    
    bnum2trialstart_nbase, bnum2trialstart_bbase : dicts from behavioral 
        trial number to behavioral trial start time, in either time base. 
        Returns None if the behavioral trial did not actually occur during 
        the recording
        Would like to make these accept lists of trial numbers ...
    
    n2b, b2n : convert time base
    
    Preferred writer methods (to be used by syncing functions)
    write_btrial_numbers : etc.
    """
    def __init__(self, rs):
        self.rs = rs
        self._set_sync()
        self._set_btrial_numbers()
        self._set_btrial_start_times()
        self._set_bnum2trialstart_d()
    
    def _set_btrial_numbers(self):
        """Defines behavioral trial numbers that actually occurred
        
        1) Those listed in TRIAL_NUMBERS
        2) If TRIAL_NUMBERS doesn't exist, load from events structure
        3) If events structure doesn't exist, load from bcontrol
        """
        try:
            self.btrial_numbers = np.loadtxt(
                os.path.join(self.rs.full_path, 'TRIAL_NUMBERS'), dtype=np.int)
        except IOError:
            # Load from bcontrol in case syncing hasn't happened yet
            1/0
    
    def _set_sync(self):
        """Defines syncing function between timebases
        
        1) Load from SYNC_B2N in RS
        """
        try:
            self._sync_poly = np.loadtxt(
                os.path.join(self.rs.full_path, 'SYNC_B2N'), dtype=np.float)
        except IOError:
            # Auto sync? Leave None?
            1/0
    
        self._sync_poly_inv = np.array([1., -self._sync_poly[1]]) \
            / self._sync_poly[0]
        
    def b2n(self, bt):
        return np.polyval(self._sync_poly, bt)
    
    def n2b(self, nt):
        return np.polyval(self._sync_poly_inv, nt)
    
    def _set_btrial_start_times(self):
        """Defines btrial onset times
        
        Requires btrial_numbers to exist, because that's how I know which
        trials actually occurred during the recording.
        
        1) As listed in events structure - which is defined to be in the
            neural time base.
        2) If doesn't exist, load from bcontrol
        
        Needs sync info, because
        Stores trialstart_nbase and trialstart_bbase
        """
        try:
            trials_info = kkpandas.io.load_trials_info(self.rs.full_path)
            events = kkpandas.io.load_events(self.rs.full_path)
        except IOError:
            # These haven't been dumped yet
            # Should load them from bcontrol file
            # Then the syncing functions can use this same accessor method
            # Code not written yet
            1/0
        
        res = []
        for bn in self.btrial_numbers:
            # Will error here if non-unique results
            res.append(events[events.event == 'trial_%d_out' % bn].time.item())
        
        self.trialstart_nbase = np.array(res)
        self.trialstart_bbase = self.n2b(self.trialstart_nbase)
    
    def _set_bnum2trialstart_d(self):
        self.bnum2trialstart_nbase = collections.defaultdict(lambda: None)
        self.bnum2trialstart_bbase = collections.defaultdict(lambda: None)
        
        assert len(self.btrial_numbers) == len(self.trialstart_nbase)
        self.bnum2trialstart_nbase.update(zip(
            self.btrial_numbers, self.trialstart_nbase))
        
        assert len(self.btrial_numbers) == len(self.trialstart_bbase)
        self.bnum2trialstart_bbase.update(zip(
            self.btrial_numbers, self.trialstart_bbase))