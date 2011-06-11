import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

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
        self.range = None
    
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
        # Get spike times to include in PSTH
        spike_times = self.pick_spikes(pick_units, pick_trials, adjusted=True)
        
        # How many trials included in this PSTH?
        if pick_trials is None:
            n_trials = self.get_number_of_trials()
        else:
            n_trials = len(pick_trials)
        
        # Initialize the PSTH
        kargs['n_trials'] = len(pick_trials)
        kargs['F_SAMP'] = self.F_SAMP
        kargs['range'] = self.range
        return PSTH(spike_times, **kargs)
    
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
        
        self.range = (-pre_win, post_win)


class PSTH(object):
    """An object providing simple peri-stimulus time histogram functions.
    
    Essentially this is a wrapper around numpy.histogram providing some
    commonly used methods and avoiding some pitfalls.
    
    Given a train of spikes, each assigned to a different trial, we wish
    to calculate some average values over those trials. You must provide
    the spike times relative to some trigger in the trial from which
    they came. Usually this trigger is the stimulus onset, so spikes may
    have negative times. But it could be any event over which you wish
    to calculate time-locked activity.
    
    Trials in a PSTH should be `homeogeneous` in the sense that you want
    to average them together. If you wish to compare two types of trials,
    create two PSTH objects and compare the results directly.
    
    When possible, values can be returned in these units:
    'spikes' : Spikes per trial per bin
    'Hz' or 'hz' : Spikes per trial per second
    
    Provided methods
    ----------------
    hist_values : A histogram of spike times relative to trigger times.
    plot : plot the histogram into some axis
    __add__ : combine PSTHs, not currently working
    time_slice : response during an epoch of time in each trial
    spont_rate : response during an epoch of time from trial start to trigger
    
    
    TODO
    ----
    Limit the number of places in which normalization to spontaneous rate
    and conversion to different units occurs, since this is a common bug.
    Probably this should happen in hist_values() only.
    
    Allow initialization from a list of trials, each with spikes. Then the user
    can still access the original per-trial information if needed.
    
    Change default behavior from norm_to_spont=True to norm_to_spont=False
    
    Keep track of spike counts and times as integers whenever possible to
    avoid floating point error.
    
    Allow recalculation when nbins or range changes.
    """
    def __init__(self, adjusted_spike_times=[], n_trials=0, 
        F_SAMP=30000., nbins=100, range=None):
        """Initialize a new PSTH.
        
        You must have already chosen your trials to include, and triggered
        the spike time and assigned them to some trial.
        
        adjust_spike_times : array of triggered spike times
        n_trials : number of trials that went into this PSTH, used for
            normalizing response to `per trial`
        F_SAMP : sampling rate, used for conversion to Hz
        nbins : Temporal resolution of the PSTH
        range : Extent of the PSTH. If you wish to compare PSTH objects,
            they should have the same extent, because this also affects
            the temporal resolution. If you don't specify it, then the minimum
            and maximum spike times will be used, but for low firing rates
            this could introduce inaccuracies!
        """
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
    
    def hist_values(self, units='spikes'):
        """Returns the histogram values as (t, counts)"""
        if units is 'spikes':
            return (self._t, self._counts / float(self.n_trials))
        elif (units is 'hz' or units is 'Hz'):
            yval = self._counts / float(self.n_trials) / self.bin_width()
            return (self._t, yval) 
       
    def plot(self, ax=None):
        """Plot PSTH into axis `ax`"""
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
    
    def time_slice(self, epoch, norm_to_spont=True, units='spikes'):
        """Return total count in epoch specified by inclusive tuple of bins.
        
        Note that because it is inclusive (unlike python indexing!), you
        have a 'fencepost problem': an epoch of (102, 110) will include
        9 bins and 9 * self.bin_width() seconds of time.
        
        If norm_to_spont, the number of spikes expected from the signal
        where self._t < 0 is subtracted from the returned value.
        """
        # Count the number of bins and the time included in them
        n_bins = epoch[1] - epoch[0] + 1
        t_bins = n_bins * self.bin_width()
        
        # Count the number of spikes
        n = self._counts[epoch[0]:epoch[1]+1].sum()
                
        if norm_to_spont:
            # Subtract off the number of expected spikes, which is the
            # spontaneous firing rate * amount of time * number of trials.
            n = n - self.spont_rate(units='hz') * t_bins * self.n_trials

        # Return in the requested units
        val_in_spikes = n / float(self.n_trials)
        if units is 'spikes':
            return val_in_spikes
        elif (units is 'hz') or (units is 'Hz'):
            return val_in_spikes / t_bins
    
    def closest_bin(self, t):
        """Returns the index of the bin whose time is closest to `t`."""
        return np.argmin(np.abs(self._t - t))
    
    def spont_rate(self, units='hz'):
        """Find the spike rate in the spontaneous regime.
        
        This regime is defined as (0, self.closest_bin(0.)), ie, from the
        first sample to the sample closest to 0 inclusive.
        """
        # We must use norm_to_spont=False here to avoid infinite loop!
        epoch = (0, self.closest_bin(0.))
        return self.time_slice(epoch=epoch, norm_to_spont=False, units=units)
    
    def bin_width(self):
        """Returns width of a single bin in seconds.
        
        To convert number of spikes in a range to spike rate, divide by
        self.bin_width() * the number of bins.
        """
        d = np.diff(self._t)
        
        # Floating point
        assert (d.max() - d.min()) < 1e-5, "t-values are irregular"
        return np.median(d)