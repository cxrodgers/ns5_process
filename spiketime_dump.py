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


import OpenElectrophy as OE
import numpy as np
import os.path
import matplotlib.mlab as mlab
import glob



def get_tetrode_block_id():
    id_blocks, = OE.sql('select block.id from block where \
        block.name = "CAR Tetrode Data"')
    return id_blocks[0]

class KlustaKwikIO(object):
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
    
def get_all_spike_times(neuron_list):
    # Some alternative code for ipython interactive psths
    big_spiketimes = dict()
    for n in neuron_list:
        stt = n._spiketrains
        spike_time_list = []
        for sttn in stt:            
            if len(sttn.spike_times) > 0:
                spike_time_list.append(sttn.spike_times - sttn.t_start)
        print n.name
        print len(spike_time_list)
        big_spiketimes[n.name] = np.concatenate(spike_time_list)
    
    return big_spiketimes

def plot_all_PSTHs(data_dir):
    # Location of OE db
    db_filename = glob.glob(os.path.join(data_dir, '*.db'))[0]
    db = OE.open_db('sqlite:///%s' % db_filename)
    
    # Get list of neurons from tetrode block
    id_block = get_tetrode_block_id()
    block = OE.Block().load(id_block)    
    neuron_list = block._neurons    
    big_spiketimes = get_all_spike_times(neuron_list)
    
    # Plot PSTHs
    for n_name, spike_time_list in big_spiketimes.items():
        plt.figure()
        plt.hist(spike_time_list, bins=100)
        #plt.savefig('test%s.png' % n_name)
        plt.show()
        


def execute(data_dir, PRE_STIMULUS_TIME=0.5):
    """Write spike time data and metadata from OE db.
    
    Writes spike times from all sorted spikes in OE db to KlustaKwik
    files in same directory.
    
    Also writes a file metadata.csv with the ntrial number (ordering
    of segments in OE db), the btrial number (trial id in matlab struct,
    extracted from OE db info field), and the time of stimulus onset.
    
    Time of stimulus onset is calculated as the t_start time of that segment
    in  OE dB, plus the provided PRE_STIMULUS_TIME parameter. Later scripts
    will use the stimulus onset times to find spikes associated with that
    trial in the KlustaKwik files.
    """
    # Location of data
    #data_dir = '/home/chris/Public/20110517_CR12B_FF2B/CR12B_0514_001'
    #PRE_STIMULUS_TIME = 0.5

    # Location of the Bcontrol file
    #bdata_filename = os.path.join(data_dir, 
    #    'data_@TwoAltChoice_v2_chris_AIR11A_101012a.at')

    # Location of OE db
    #db_filename = os.path.join(data_dir, 'datafile_CR_CR12B_110507_001.db')
    db_filename = glob.glob(os.path.join(data_dir, '*.db'))[0]

    # Output of the writers
    output_filename = os.path.splitext(db_filename)[0]
    
    # Metadata: trial numbers etc
    metadata_filename = os.path.join(data_dir, 'metadata.csv')

    # Load db
    db = OE.open_db('sqlite:///%s' % db_filename)
    
    # Get list of segments from tetrode block
    id_block = 2
    block = OE.Block().load(id_block)    
    neuron_list = block._neurons
    

    
    # Build a writer
    w = KlustaKwikIO(filename=output_filename)    
    w.write_block(block)
    
    # Also dump metadata: btrial num, t_start in samples
    seg_metadata = list()
    for seg in block.get_segments():
        t_starts1 = [sig.t_start for sig in seg._analogsignals]
        t_starts2 = [st.t_start for st in seg._spiketrains]
        assert len(np.unique(t_starts1)) == 1
        
        # You can get errors here where some spiketrains have already the right
        # t_start and others don't. I think maybe this happens when you re-spike
        # sort or something.
        #assert len(np.unique(t_starts2)) == 1
        #assert np.unique(t_starts1) == np.unique(t_starts2)
        
        
        # This was a stupid bugfix for a stupid bug that is now breaking things
        # Replaced with PRE_STIMULUS_TIME so at least it's up front
        t_start = np.rint((t_starts1[0] + PRE_STIMULUS_TIME) * 30000.).astype(np.int64)        
        
        seg_metadata.append((seg.name, int(seg.info), t_start))
    
    # Convert seg_metadata to recarray and write to disk
    r = np.rec.fromrecords(seg_metadata,
        dtype=[('ntrial','<U32' ), ('btrial_num', np.int), ('stim_onset', np.int64)])
    mlab.rec2csv(r, metadata_filename)
    