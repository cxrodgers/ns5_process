import numpy as np
import os.path
import glob
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import bcontrol
import pickle
from collections import defaultdict


def ismember(ar1, ar2):
    # This is the inner loop in spike picking (.7s per call!)
    # Replaced with dicts.
    return np.asarray([val in np.asarray(ar2) for val in np.asarray(ar1)])

def only_one(x):
    try:
        assert len(x) == 1        
        val = x[0]    
    except TypeError:
        val = x    
    return val

class MultipleUnitSpikeTrain(object):
    """Simple container for yoked spike_times and unit_IDs"""
    def __init__(self, spike_times, unit_IDs=None, spike_trials=None, 
        peri_onset_spike_times=None):
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
    
    def _pick_spikes_mask_old(self, pick_units=None, pick_trials=None):
        """Returns a mask of spike_times for specified trials and units.
        Deprecated, slow.
        """
        if pick_units is None:
            pick_units = self.get_unique_unit_IDs()
        if pick_trials is None:
            pick_trials = self.spike_trials
        mask = \
            ismember(self.unit_IDs, pick_units) & \
            ismember(self.spike_trials, pick_trials)
        return mask
    
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
        
        # Build an onset-relative representation
        self.peri_onset_spike_times = self.spike_times - \
            onsets[spike_trials_idx]
        
        self._build_dicts()

class KK_loader(object):
    """Loads spike info from KlustaKwik-compatible files."""
    def __init__(self, data_dir):
        """Initializes a new KK_loader, linked to the data directory.
        
        data_dir : string, path to location of KlustaKwik files.
        """
        self._data_dir = data_dir

    def execute(self):
        """Loads spike times and unit info from KlustaKwik files.
        
        Stores resulting spike trains in self.spiketrains, a dict keyed
        by tetrode number.
        """
        # Search data directory for KlustaKwik files. This sets
        # self._clufiles and self._fetfiles.
        self._get_fns_from_datadir()
        
        # Load spike times from each tetrode and store in a dict, keyed
        # by tetrode number
        self.spiketrains = dict()
        for ntet in self._fetfiles.keys():
            fetfile = self._fetfiles[ntet]
            clufile = self._clufiles[ntet]            
            spks = self._load_spike_times(fetfile)
            uids = self._load_unit_id(clufile)
            
            # Initialize a new container for the loaded spike times and IDs
            self.spiketrains[ntet] = MultipleUnitSpikeTrain(spks, uids)
    
    def _get_fns_from_datadir(self):
        """Stores dicts of KlustaKwik *.clu and *.fet files from directory."""
        self._clufiles = self._parse_KK_files(\
            glob.glob(os.path.join(self._data_dir, '*.clu.*')), 'clu')
        self._fetfiles = self._parse_KK_files(\
            glob.glob(os.path.join(self._data_dir, '*.fet.*')), 'fet')

    def _parse_KK_files(self, filename_list, match_string='fet'):
        """Returns a dict of filenames keyed on tetrode number.
        
        You give a list of filenames, obtained by globbing, that are either
        cluster or feature files. You also specify whether they are `clu`
        or `fet` files.
        
        This matches on tetrode number and returns a dict of the filenames
        keyed by tetrode number.
        """
        return dict([\
            (int(glob.re.search(('%s\.(\d+)' % match_string), v).group(1)), v) \
            for v in filename_list])

    def _load_spike_times(self, fetfilename):
        f = file(fetfilename, 'r')
        
        # Number of clustering features is integer on first line
        nbFeatures = int(f.readline().strip())
        
        # Each subsequent line consists of nbFeatures values, followed by
        # the spike time in samples.
        names = ['feat%d' % n for n in xrange(nbFeatures)]
        names.append('spike_time')
        
        # Load into recarray
        data = mlab.csv2rec(f, names=names, skiprows=1)
        f.close()
        
        # Return the spike_time column
        return data['spike_time']

    def _load_unit_id(self, clufilename):
        f = file(clufilename, 'r')
        
        # Number of clusters on this tetrode is integer on first line
        nbClusters = int(f.readline().strip())
        
        # Each subquent line is a cluster ID (string)
        cluster_names = f.readlines()
        f.close()

        # Extract the number of the OE neuron name        
        #~ cluster_ids = np.array([\
            #~ int(glob.re.match('Neuron (\d+) ', name).group(1)) \
            #~ for name in cluster_names])
        cluster_ids = np.zeros((len(cluster_names),))
        for n, name in enumerate(cluster_names):
            m = glob.re.match('Neuron (\d+) ', name)
            if m is not None:
                cluster_ids[n] = int(m.group(1))
            else:
                cluster_ids[n] = 99

        # Simple error checking
        assert(len(np.unique(cluster_ids)) == nbClusters)
        assert(len(np.unique(cluster_names)) == nbClusters)

        return cluster_ids


def calc_psth(st):   
    if len(st) == 0:
        return (np.array([]), np.array([]))
    h, bin_edges = np.histogram(st, bins=100)
    bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
    return (h, bin_centers/30000.)


class PSTH(object):
    def __init__(self, adjusted_spike_times=[], n_trials=0, 
        F_SAMP=30000., nbins=100):
        # Store parameters and data
        self.F_SAMP = F_SAMP
        self.nbins = nbins
        self.n_trials = n_trials
        self.adjusted_spike_times = adjusted_spike_times
        
        # Bin and store _counts and _t
        self._calc_psth()
    
    def _calc_psth(self):
        if len(self.adjusted_spike_times) == 0:
            self._t = np.array([])
            self._counts = np.array([])
        else:
            self._counts, bin_edges = np.histogram(self.adjusted_spike_times,
                self.nbins)
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
        #plt.xticks(np.linspace(self._t.min(), self._t.max(), 4))
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


def MUA_PSTH_vs_sn2(spiketrain, sn2trials, sn2name, nbins=300):
    tet_psth_vs_sn = dict()
    for sn, trial_list in sn2trials.items():
        keepspikes = spiketrain.pick_spikes(pick_trials=trial_list)
        tet_psth_vs_sn[sn] = PSTH(keepspikes, len(trial_list), nbins=nbins)
    return tet_psth_vs_sn    

def SUA_PSTH_vs_sn2(spiketrain, uid, sn2trials, sn2name, nbins=300):
    uu_psth_vs_sn = dict()
    for sn, trial_list in sn2trials.items():
        keepspikes = spiketrain.pick_spikes(pick_trials=trial_list, pick_units=[uid])
        uu_psth_vs_sn[sn] = PSTH(keepspikes, len(trial_list), nbins=nbins)
    return uu_psth_vs_sn    
    

def MUA_PSTH_vs_sn(spiketrain, sn2trials, sn2name, savename=None):
    """Given spiketrain from tetrode and mappings of stimuli names
    and trials, plots an MUA PSTH."""
    plt.figure()            
    #plt.suptitle('T%d:MUA' % (tetn,))    
    for sn, trial_list in sn2trials.items():
        keepspikes = spiketrain.pick_spikes(pick_trials=trial_list)
        h, t = calc_psth(keepspikes)
        plt.subplot(3, 4, sn)
        plt.plot(t, h / float(len(trial_list)))
        plt.title(sn2name[sn])

    plt.show()
    
    if savename is not None:
        plt.savefig(savename)

    

def get_trial_numbers_vs_sn(TRIALS_INFO, CONSTS):
    trial_numbers_vs_sn = dict()
    for sn in np.unique(TRIALS_INFO['STIM_NUMBER']):
        keep_rows = \
            (TRIALS_INFO['STIM_NUMBER'] == sn) & \
            (TRIALS_INFO['OUTCOME'] == CONSTS['HIT']) & \
            (TRIALS_INFO['NONRANDOM'] == 0)        
        trial_numbers_vs_sn[sn] = TRIALS_INFO['TRIAL_NUMBER'][keep_rows]
    return trial_numbers_vs_sn    


def get_bdata_pickle(data_dir):
    # Get behavioral data, pickling as necessary
    # Move this into bcontrol.py
    try:
        fn_pickle = only_one(glob.glob(os.path.join(data_dir, 'bdata.pickle')))
    except AssertionError:
        # No pickle, must load the slow way
        fn_bdata = only_one(glob.glob(os.path.join(data_dir, 'data_*.mat')))        
        bcl = bcontrol.Bcontrol_Loader(filename=fn_bdata, v2_behavior=False)
        bcl.load()
        
        # Store in pickle for next time
        fn_pickle = os.path.join(data_dir, 'bdata.pickle')
        pickle.dump(bcl.data, file(fn_pickle, 'w'))
    
    # Load (or re-load) the data from the pickle
    f = file(fn_pickle, 'r')
    bcl_data = pickle.load(f)
    f.close()
    
    return bcl_data



def execute(data_dir):
    # BEGIN MAIN SCRIPT HERE
    # Where the files are located
    #~ unit_map = [\
        #~ ('003', 0, [0,1,2,3]),
        #~ ('003', 1, [0,1,2,3]),
        #~ ('003', 2, [0,1,2,3]),
        #~ ('003', 3, [0,1,2,3]),
        #~ ('003', 4, [0,1,2,3]),
        #~ ('003', 5, [0,1,2,3]),

    
    #~ save_figs = True8
            
    #~ for uu in unit_map:
        #~ # Load data from this directory
    #data_dir = '/home/chris/Public/20110517_CR12B_FF2B/CR12B_0514_001'


    # Initialize a KlustKwik loader and run it on the data directory.
    # Spike trains will then be stored in kkl.spiketrains
    kkl = KK_loader(data_dir)
    kkl.execute()
    #assert(len(kkl.spiketrains) == 4) # Check I didn't forget to sort a tetrode

    # Get metadata, which translates neural trial numbers into behavioral,
    # and also provides pre-calculated stimulus onsets.
    fn_metadata = only_one(glob.glob(os.path.join(data_dir, 'metadata.csv')))
    metadata = mlab.csv2rec(fn_metadata)

    # Get behavioral files
    bcl_data = get_bdata_pickle(data_dir)
    trials_info = bcl_data['TRIALS_INFO']
    consts = bcl_data['CONSTS']

    # Construct dict of stimulus numbers and btrial numbers
    sn2trials = get_trial_numbers_vs_sn(trials_info, consts)
    sn2name = dict([(n+1, sndname) for n, sndname in \
        enumerate(bcl_data['SOUNDS_INFO']['sound_name'])])

    # Calculate each tetrode's data separately
    todays_psths = dict()
    todays_sorted_psths = dict()
    for tetn, spiketrain in kkl.spiketrains.items():
        todays_sorted_psths[tetn] = dict()
        
        # Link the spike times to the trials from whence they came
        spiketrain.add_trial_info(metadata['stim_onset'], metadata['btrial_num'],
        pre_win=45000, post_win=45000)
        
        
        # Plot MUA PSTH of each tetrode
        #~ todays_psths[tetn] = MUA_PSTH_vs_sn2(spiketrain, sn2trials, sn2name,
            #~ nbins=300)
        #~ plt.figure()        
        #~ for sn in todays_psths[tetn].keys():
            #~ ax = plt.subplot(3, 4, sn)
            #~ todays_psths[tetn][sn].plot(ax=ax)
            #~ plt.title(sn2name[sn])
        #~ plt.suptitle('MUA PSTH on tet %d' % tetn)
        #~ plt.show()
        
        # Now go through sorted units
        for uid in spiketrain.get_unique_unit_IDs():
            todays_sorted_psths[tetn][uid] = \
                SUA_PSTH_vs_sn2(spiketrain, uid, sn2trials, sn2name, nbins=300)
            
            
            # And plot
            plt.figure()
            for sn in todays_sorted_psths[tetn][uid].keys():
                ax = plt.subplot(3, 4, sn)
                todays_sorted_psths[tetn][uid][sn].plot(ax=ax)
                plt.title(sn2name[sn])
            plt.suptitle('SU_%f PSTH on tet %d' % (uid, tetn))
            plt.show()            
        
