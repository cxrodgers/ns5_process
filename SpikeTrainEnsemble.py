NS5_PROCESS_PATH = '/home/chris/ns5_process'
import sys
# Path to ns5_process
if NS5_PROCESS_PATH not in sys.path:
    sys.path.append(NS5_PROCESS_PATH)

import numpy as np
import bcontrol
from collections import defaultdict
import matplotlib
matplotlib.rcParams['figure.subplot.hspace'] = .5
matplotlib.rcParams['figure.subplot.wspace'] = .5
matplotlib.rcParams['font.size'] = 8.0
matplotlib.rcParams['xtick.labelsize'] = 'small'
matplotlib.rcParams['ytick.labelsize'] = 'small'
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os.path
import KlustaKwikIO
import SpikeTrainContainers


class SpikeTrainEnsemble:
    """Stores and serves spiketrains matching various parameters.
    
    Generally this will be loaded using a dedicated object.
    # Load KlustaKwik data, bcontrol data (if available), task constants
    ste = SpikeTrainEnsembleCreator(data_dir_list)
    
    Use cases:
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
        
        if self.sn2name is None:
            self.sn2name = sn2name
            
            # Generate self.name2sn here
            self.name2sn = dict()
            for k, v in self.sn2name.items():
                self.name2sn[v] = k
            
            assert(len(self.name2sn) == len(self.sn2name)), 'key collision'
            
        else:
            # Verify that sn2name has not changed here
            pass
       
    def pick_spikes_from_block_list(self, block_list=None, stim=None, **kargs):
        """Assume you want all tetrodes and return all spike times"""
        if block_list is None:
            block_list = self._list_blocks
        
        spike_list = list()
        for block in block_list:
            idx_block = self._list_blocks.index(block)
            
            # Find stim number
            try:
                self.sn2name[stim]
            except KeyError:
                # The stim number was not provided, must be stim name
                stim = self.name2sn[stim]
            
            # Find trial indexes using what I know about sn2trials
            musts = self._list_musts[idx_block]
            if stim is not None:
                trial_list = self._list_sn2trials[idx_block][stim]
            else:
                trial_list = None     
            
            # Iterate through tet
            for tetnum in sorted(musts.keys()):
                must = musts[tetnum]
                spikes = must.pick_spikes(pick_trials=trial_list, 
                    adjusted=True, **kargs)
                spike_list.append(spikes)
        
        return np.concatenate(spike_list)
    
    def get_psth_from_block_list(self, block_list=None, stim=None, with_labels=False, **kargs):
        """Assume you want all tetrodes and return everything."""
        if block_list is None:
            # Use all
            block_list = self._list_blocks
        
        # Choose list of stimuli to use
        if stim is None:
            # Use all
            stim = self.sn2name.keys()
        elif not np.iterable(stim):
            # Probably a single int
            stim = [stim]
        elif isinstance(stim, str):
            # Probably a single string
            stim = [stim]
        
        # Convert to stim number
        stim_list = list()
        for s in stim:
            try:
                # Try to append sn for the given name
                stim_list.append(self.name2sn[s])
            except KeyError:
                # Not a stim name, assume it's a number
                stim_list.append(s)        
        
        # Get psth for each block
        psth_list = list()
        labels = list()
        for block in block_list:
            # Get spikes
            idx_block = self._list_blocks.index(block)
            
            # Find trial indexes using what I know about sn2trials
            musts = self._list_musts[idx_block]
            trial_list = np.concatenate(\
                [self._list_sn2trials[idx_block][s] for s in stim_list])
            
            # Iterate through tet
            for tetnum in sorted(musts.keys()):
                must = musts[tetnum]
                psth = must.get_psth(pick_trials=trial_list, **kargs)
                labels.append('block %s tetnum %d' % (block, tetnum))
                psth_list.append(psth)
        
        if with_labels:
            return psth_list, labels
        else:
            return psth_list
    
    def get_spike_count_from_block_list2(self, timeslice, block=None, sn=[], 
        norm_to_spont=False, with_labels=False, units='spikes'):
        # Get all psths from every tetrode and every stim
        all_psth = np.array([self.get_psth_from_block_list(block, s) for s in sn])
        
        # Find regime for each
        regimes = np.array([[[p.closest_bin(t) for t in timeslice] \
            for p in pl] for pl in all_psth])
        
        # Find resp for each
        resps = np.array([[p.time_slice(regime, norm_to_spont, units=units) \
            for p, regime in zip(pl, regl)] \
            for pl, regl in zip(all_psth, regimes)])
        
        # Average across stimuli
        mean_resps = resps.mean(axis=0)

        # Return with labels if requested
        if with_labels:
            labels = self.get_psth_from_block_list(block, sn[0], with_labels=True)[1]
            return mean_resps, labels
        else:
            return mean_resps
        
    def plot_all_stimuli_psths(block_list=None, bins=300):
        plt.figure()
        for k, v in self.sn2name.items():
            ax = plt.subplot(3, 4, k)
            psth_yval_list = list()
            tval_keep = None
            for psth in ste.get_psth_from_block_list(block_list=block_list,
                stim=[v], nbins=bins):
                tval, yval = psth.hist_values(units='spikes')
                psth_yval_list.append(yval)
                if tval_keep is None:
                    tval_keep = tval
                else:
                    assert np.all(tval_keep == tval), 'inconsistent t-values in PSTHs'
            
            plt.plot(tval, np.mean(np.array(psth_yval_list), axis=0))
            plt.ylim((0, 1))
            plt.title(v)        







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
        if self.ste is None:
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