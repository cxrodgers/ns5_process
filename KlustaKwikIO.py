# Iterate through all neurons in OE db and dump spiketimes
# We will use the KlustaKwik compatible Feature File format
# Ref: http://klusters.sourceforge.net/UserManual/data-files.html
# Begin format specification (lightly modified from above):
"""
The Feature File

Generic file name: base.fet.n

Format: ASCII, integer values

The feature file lists for each spike the PCA coefficients for each
electrode, followed by the timestamp of the spike (more features can
be inserted between the PCA coefficients and the timestamp). 
The first line contains the number of dimensions. 
Assuming N1 spikes (spike1...spikeN1), N2 electrodes (e1...eN2) and
N3 coefficients (c1...cN3), this file looks like:

nbDimensions
c1_e1_spike1   c2_e1_spike1  ... cN3_e1_spike1   c1_e2_spike1  ... cN3_eN2_spike1   timestamp_spike1
c1_e1_spike2   c2_e1_spike2  ... cN3_e1_spike2   c1_e2_spike2  ... cN3_eN2_spike2   timestamp_spike2
...
c1_e1_spikeN1  c2_e1_spikeN1 ... cN3_e1_spikeN1  c1_e2_spikeN1 ... cN3_eN2_spikeN1  timestamp_spikeN1

The timestamp is expressed in multiples of the sampling interval. For
instance, for a 20kHz recording (50 microsecond sampling interval), a
timestamp of 200 corresponds to 200x0.000050s=0.01s from the beginning
of the recording session.

Notice that the last line must end with a newline or carriage return. 
"""

import numpy as np
import glob
import matplotlib.mlab as mlab
from SpikeTrainContainers import MultipleUnitSpikeTrain
import os.path


class KlustaKwikIO(object):
    """TODO: fix for case where spikes are not sorted and no neurons exist."""
    def __init__(self, filename=None):
        self.filename = filename
        self._fetfiles = dict()
        self._clufiles = dict()
    
    def _new_group(self, id_group, nbClusters):
        self._fetfiles[id_group] = file(self.filename + \
            ('.fet.%d' % id_group), 'w')
        self._clufiles[id_group] = file(self.filename + \
            ('.clu.%d' % id_group), 'w')
        
        self._fetfiles[id_group].write("0\n") # Number of features
        self._clufiles[id_group].write("%d\n" % nbClusters)
    
    def _get_file_handles(self, st):
        try:
            id_group = st.recordingpoint.group
        except AttributeError:
            id_group = -1
        
        try:
            fetfile = self._fetfiles[id_group]
            clufile = self._clufiles[id_group]
        except KeyError:
            self._new_group(id_group, 0)
            fetfile = self._fetfiles[id_group]
            clufile = self._clufiles[id_group]
        
        return (fetfile, clufile)
    
    def _close_all_files(self):
        for val in self._fetfiles.values():
            val.close()
        
        for val in self._clufiles.values():
            val.close()
    
    def _make_all_file_handles(self, neuron_list):
        # Get the tetrode (group) of each neuron (cluster)
        id_groups = list()
        for n in neuron_list:
            try:
                id_group = n._spiketrains[0].recordingpoint.group
            except (IndexError, AttributeError):
                id_group = -1
            
            if id_group is None:
                id_group = n._spiketrains[0].recordingpoint.id
            
            id_groups.append(id_group)

        # Make new file handles for each group
        for id_group in np.unique(id_groups):
            self._new_group(id_group, nbClusters=id_groups.count(id_group))        
    
    def write_block(self, block):     
        # Make all file handles
        neuron_list = block._neurons
        #1/0
        self._make_all_file_handles(neuron_list)
        
        # Will contain info about each segment
        segment_info = list()
        
        # Iterate through segments in this block
        for seg in block.get_segments():
            # Write each spiketrain of the segment
            for st in seg.get_spiketrains():  
                # Get file handles for this spiketrain using its group
                fetfile, clufile = self._get_file_handles(st)
                
                # Write each spike of the spiketrain                
                for stt in np.rint(st.spike_times * 30e3).astype(int):
                    # Would normally write features here
                    # But currently features are not stored
                    
                    # Write spike time in samples
                    fetfile.write("%d\n" % stt)
                    
                    # Write neuron id
                    clufile.write("%s\n" % st.name)
        
        self._close_all_files()

class KK_loader(object):
    """Loads spike info from KlustaKwik-compatible files.
    
    TODO: fix for case when data is unsorted (no Neuron in block), currently
        something to do with 99
    TODO: handle trash somehow
    TODO: store neuron names
    """
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
        
        It must end with digits, not including minus sign or anything else.
        """
        d = dict()
        for v in filename_list:
            # Test whether matches format, ie ends with digits
            m = glob.re.search(('%s\.(\d+)' % match_string), v)             
            if m is not None:
                # Key the tetrode number to the filename
                tetn = int(m.group(1)) # Group 0 is something else
                d[tetn] = v
        return d
        
        #~ return dict([\
            #~ (int(glob.re.search(('%s\.(\d+)' % match_string), v).group(1)), v) \
            #~ for v in filename_list])

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

