NS5_PROCESS_PATH = '/home/chris/ns5_process'
import sys
# Path to ns5_process
if NS5_PROCESS_PATH not in sys.path:
    sys.path.append(NS5_PROCESS_PATH)

import numpy as np
import bcontrol
from collections import defaultdict
import matplotlib.mlab as mlab
import os.path
import KlustaKwikIO


class SpikeTrainEnsemble:
    """Stores and serves spiketrains matching various parameters.
    
    # Load KlustaKwik data, bcontrol data (if available), task constants
    ste = SpikeTrainEnsembleCreator(data_dir_list)
    
    # PSTH on stim from all dates
    # The PSTH object can't handle multiple dates, so your only option is
    # to get spikes explicitly (or add PSTHs together)
    # PSTH object holds only "homogeneous" data
    # Splittable data goes into STE
    # MUST is linked to a single tetrode
    plt.hist(ste.pick_spikes(stim='lelolc'))
    
    # PSTH on each stim on certain date
    for stim in ste.sn2name.values():
        psth = ste.get_psth(date='1001', stim=stim)
        psth.plot()
    
    # Get strengths of response in each spiketrain
    strengths, idxs = ste.get_spike_count(\
        t_start, t_stop, norm_to_spont=True, stim=['lelolc', 'lehilc'])
    
    
    Maybe also create a wrapper than handles translating stimulus numbers
    into trial numbers.
    """
    def __init__(self):
        """New SpikeTrainEnsemble
        For right now all data is stored in lists with one entry per data_dir
        
        _list_blocks : Each entry is name of that block (data_dir)
        _list_musts : Each entry is a dict of MultipleUnitSpikeTrain from `date`,
            keyed on tetrode number
        _list_sn2trials : Each entry is sn2trials for that date
            This is just a convenience wrapper for pick_trials
        
        
        All must use these conventions
        sn2name : dict of stimulus number to stimulus name
        name2sn : vice versa
        
        Methods available
        pick_spikes : raw concatenated spike times
        get_psth : get PSTH object matching conditions
        get_spike_count : count spikes within range on data matching certain
            conditions
        """
        self._list_musts = list()
        self._list_blocks = list()
        self._list_sn2trials = list()
        
        self.sn2name = None
        self.name2sn = None
    
    def add_block(self, musts, block_name, sn2trials, sn2name):
        self._list_musts.append(musts)
        self._list_blocks.append(block_name)
        self._list_sn2trials.append(sn2trials)
        
        if self.sn2name is not None:
            self.sn2name = sn2name
            # Generate self.name2sn here
        
        else:
            # Verify that sn2name has not changed here
            pass
    
    def get_psth(self, block=None, sn=None, range=None):
        trial_list = self._list_sn2trials(idx)[sn]
        keepspikes = spiketrain.pick_spikes(pick_trials=trial_list, 
            pick_units=[])
        return PSTH(keepspikes, len(trial_list), nbins=nbins, range=range)
    
    def get_spike_count(self, timeslice, block=None, sn=None, norm_to_spont=False):
        # Eventually count spikes here, or make PSTH count with higher
        # temporal resolution.
        self.get_psth(block, sn).spike_count(timeslice, norm_to_spont)












class SpikeTrainEnsembleCreator:
    """Idiosyncratic class for loading bcontrol metadata and neural data"""
    def __init__(self, pre_win, post_win):
        self.ste = None
        self.pre_win = pre_win
        self.post_win = post_win
    
    def load_list_of_dirs(self, data_dir_list, name_list, verbose=True):
        # Create a new SpikeTrainEnsemble to return
        self.ste = SpikeTrainEnsemble()
        
        # Iterate through directories and load each
        for data_dir, name in zip(data_dir_list, name_list):
            if verbose:
                print data_dir
                sys.stdout.flush()
            self.load_single_dir(data_dir, name)
        
        return self.ste
    
    def load_single_dir(self, data_dir, name):
        # Create a new SpikeTrainEnsemble to return
        if self.ste is not None:
            self.ste = SpikeTrainEnsemble()
        
        # Load spike info
        kkl = KlustaKwikIO.KK_loader(data_dir)
        kkl.execute()
        fn_metadata = os.path.join(data_dir, 'metadata.csv')
        metadata = mlab.csv2rec(fn_metadata)
        
        # Instantiate and run an object to load bcontrol data from data_dir
        bcontrol_loader = bcontrol.Bcontrol_Loader_By_Dir(data_dir)
        bcontrol_loader.load()
        sn2trials = bcontrol_loader.get_sn2trials()
        sn2name = bcontrol_loader.get_sn2name()
        
        # Add metadata to each spiketrain and add to today
        for tetnum, must in kkl.spiketrains.items():
            must.add_trial_info(metadata['stim_onset'], 
                metadata['btrial_num'], pre_win=self.pre_win, 
                post_win=self.post_win)        
    
        # Add today's data to population
        self.ste.add_block(kkl.spiketrains, name, sn2trials, sn2name)
        
        return self.ste