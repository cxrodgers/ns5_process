import numpy as np
from collections import defaultdict

class MultipleUnitSpikeTrain(object):
    """Simple container for yoked spike_times and unit_IDs"""
    def __init__(self, spike_times, unit_IDs=None, spike_trials=None, 
        peri_onset_spike_times=None, F_SAMP=30000.):
        """Initializes a new spike train.
        
        spike_times : time of each spike in samples
        unit_IDs : unit to which each spike belongs
        spike_trials : trial during which each spike occurred
        peri_onset_spike_times : timing of each spike relative to the
            onset of each trial.
        
        If provided, each argument should have the same shape as spike_times.
        """
        self.spike_times = np.array(spike_times)        
        self.unit_IDs = np.array(unit_IDs)
        self.spike_trials = np.array(spike_trials)
        self.peri_onset_spike_times = np.array(peri_onset_spike_times)
        self.F_SAMP = F_SAMP
    
    def pick_spikes(self, pick_units=None, pick_trials=None, adjusted=True):
        """Returns spike times from specified units and trials.
        
        You must set self.spike_trials before calling this method.
        If adjusted=False, original spike times (rather than
        peri_onset_spike_times) will be returned.
        """
        mask = self._pick_spikes_mask(pick_units, pick_trials)
        if adjusted:
            return self.peri_onset_spike_times[mask]
        else:
            return self.spike_times[mask]
    
    def get_psth(self, pick_units=None, pick_trials=None, **kargs):
        spike_times = self.pick_spikes(pick_units, pick_trials, adjusted=True)
        n_trials = self.get_number_of_trials()
        
        kargs['n_trials'] = n_trials
        kargs['F_SAMP'] = self.F_SAMP
        return PSTH(spike_times, kargs)
    
    def _pick_spikes_mask(self, pick_units=None, pick_trials=None):
        """Returns a mask of spike_times for specified trials and units."""
        if pick_units is None:
            #pick_units = self.get_unique_unit_IDs()
            # All true
            mask1 = np.ones(self.spike_times.shape, dtype=bool)
        elif len(pick_units) == 0:
            # All false
            mask1 = np.zeros(self.spike_times.shape, dtype=bool)
        else:
            mask1 = reduce(np.logical_or, [self._id2spkmask[u] \
                for u in pick_units])
    
        if pick_trials is None:
            #pick_trials = self.spike_trials
            mask2 = np.ones(self.spike_times.shape, dtype=bool)
        elif len(pick_trials) == 0:
            mask2 = np.zeros(self.spike_times.shape, dtype=bool)
        else:
            # TODO: check this bugfix
            # Errors occur when a trial is requested is not in 
            # self._tr2spkmask.keys(), presumably because it was not detected
            # for whatever reason.
            
            mask2 = reduce(np.logical_or, [self._tr2spkmask[t] \
                for t in pick_trials])

        mask = mask1 & mask2
        return mask
    
    def _build_dicts(self):
        """Build lookup dicts used to speed up pick_spikes"""
        self._id2spkmask = dict([(id, self.unit_IDs == id) \
            for id in self.get_unique_unit_IDs()])
        
        
        # Annoying bug where trials that don't exist in self.spike_trials
        # are called in pick_trials, and cause key error.
        # Why is this happening anyway??
        # TODO: check this bugfix
        
        # Old
        #self._tr2spkmask = dict([(tr, self.spike_trials == tr) \
        #    for tr in np.unique(self.spike_trials)])
        
        # New
        self._tr2spkmask = defaultdict(\
            lambda: np.zeros(self.spike_trials.shape, dtype=np.bool))
        for tr in np.unique(self.spike_trials):
            self._tr2spkmask[tr] = (self.spike_trials == tr)
        

        #~ # Some silly hacks for missing trials, fix me!
        #~ if 2 not in self._tr2spkmask.keys():
            #~ self._tr2spkmask[2] = (self.spike_trials == 2)
        #~ if 3 not in self._tr2spkmask.keys():
            #~ self._tr2spkmask[3] = (self.spike_trials == 3)
        #~ for tr in np.arange(408, 501):
            #~ if tr not in self._tr2spkmask.keys():
                #~ self._tr2spkmask[tr] = (self.spike_trials == tr)        
            
    
    def get_unique_unit_IDs(self):
        return np.unique(self.unit_IDs)
    
    def get_number_of_units(self):
        return len(self.get_unique_unit_IDs)
    
    def get_number_of_trials(self):
        return len(np.unique(self.spike_trials))
    
    def add_trial_info(self, onsets, onset_trial_numbers, pre_win, post_win):
        """Links each spike in the spiketrain with a trial.
        
        Parameters
        ----------
        onsets : the timestamps of the trials in samples
        onset_trial_numbers : the desired numbering of the trials 
            (use behavioral). This should be the same shape as `onsets`.
        pre_win, post_win : the number of samples around each onset which
        constitutes the trial. Should be positive.
        
        Each spike will be assigned to one trial based on the epoch during
        which it occurs. A warning is issued if this would cause the spike
        to belong to more than one trial, in which case the first one is used.
        
        After this method is called, self.spike_trials will be the same length
        as self.spike_times and self.unit_IDs, and will contain the trial ID
        of each spike. Also, self.peri_onset_spike_times will be set relative
        to each spike's trial.
        """
        assert (len(onsets) == len(onset_trial_numbers))
        
        # For each spike, find the matching stimulus onset, and store this
        # index into `onsets` and `onset_trial_numbers` in `spike_trials_idx`.
        # This loop would be faster if I iterated over onsets and masked
        # the spike times.
        no_matches = 0
        extra_matches = 0
        spike_trials_idx = np.zeros_like(self.spike_times)
        for n, stt in enumerate(self.spike_times):
            matching_idx = np.where((stt - onsets < post_win) & \
                (stt - onsets > -pre_win))[0]

            if len(matching_idx) > 1:
                # More than one trial matched this spike
                extra_matches += 1

            # Assign spike_trials_idx
            try:
                spike_trials_idx[n] = matching_idx[0]
            except IndexError:
                assert len(matching_idx) == 0
                no_matches += 1
                
                # This code used to work but it doesn't anymore
                # I'm not sure how to handle this case of spikes not belonging
                # to a trial. Hopefully it shouldn't happen ...
                spike_trials_idx[n] = np.nan

        if ((no_matches > 0) or (extra_matches > 0)):
            print ("WARNING: %d spikes matched no trials and " % no_matches) + \
                ("%d matched more than one" % extra_matches)

        # Convert indexes into `onset_trial_numbers` into trial_numbers.
        self.spike_trials = onset_trial_numbers[spike_trials_idx]
        
        # Warn if not all trials were used
        # If this happens a lot, should change get_number_of_trials
        # to use length of onset_trial_numbers instead.
        if len(np.unique(self.spike_trials)) != \
            len(np.unique(onset_trial_numbers)):
            print "warning: not all trials contain spikes. " + \
                "this may throw off PSTH calculations"
        
        # Build an onset-relative representation
        self.peri_onset_spike_times = self.spike_times - \
            onsets[spike_trials_idx]
        
        self._build_dicts()


class PSTH(object):
    def __init__(self, adjusted_spike_times=[], n_trials=0, 
        F_SAMP=30000., nbins=100, range=None):
        # Store parameters and data
        self.F_SAMP = F_SAMP
        self.nbins = nbins
        self.n_trials = n_trials
        self.adjusted_spike_times = adjusted_spike_times
        self.range = range
        
        # Bin and store _counts and _t
        self._calc_psth()
    
    def _calc_psth(self):
        if len(self.adjusted_spike_times) == 0:
            self._t = np.array([])
            self._counts = np.array([])
        else:
            self._counts, bin_edges = np.histogram(self.adjusted_spike_times,
                bins=self.nbins, range=self.range)
            self._t = (bin_edges[:-1] + 0.5 * np.diff(bin_edges)) / self.F_SAMP
    
    def plot(self, ax=None):
        if len(self._t) == 0:
            return
        if ax is None:
            plt.figure()
            plt.plot(self._t, self._counts / float(self.n_trials))
        else:
            ax.plot(self._t, self._counts / float(self.n_trials))
        #plt.xlim(self._t.min(), self._t.max())
        #plt.xticks(np.linspace(self._t.min(), self._t.max(), 5))
        plt.xticks([])
        plt.yticks(np.arange(np.ceil(self._counts.max() / float(self.n_trials))+1))
    
    def __add__(self, psth2):
        p = PSTH()
        p._t = self._t
        p._counts = self._counts + psth2._counts
        p.n_trials = self.n_trials + psth2.n_trials
        return p
    
    def time_slice(self, epoch, norm_to_spont=True):
        """Return total count in epoch specified by tuple of bins"""
        n = self._counts[epoch[0]:epoch[1]+1].sum()
        if norm_to_spont:
            ns = self.spont_rate() * (epoch[1] - epoch[0] + 1) * \
                (self._t[-1] - self._t[0]) / self.nbins
        else:
            ns = 0.
        
        return (n / float(self.n_trials)) - ns
    
    def closest_bin(self, t):
        return np.argmin(np.abs(self._t - t))
    
    def spont_rate(self):
        return self.time_slice((0, self.closest_bin(0)), 
            norm_to_spont=False) / -self._t[0]