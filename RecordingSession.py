"""Helper functions and variables for all data in a recording session.

This allows you to create a well-formed directory for ns5_process and fill
it with data.

Try to keep logic relating to any specific experiment out of this module.

RecordingSession spec:
* A directory
* File containing neural channels to put into database
  16 17 18 20 22 24 26 28
* File containing channel groupings
  16 17 18 20
  22 24 26 28
* File containing analog channels to put into database (if any)
  7 8
* TIMESTAMPS with times in samples to extract
"""

import shutil
import glob
import os.path
import ns5
import numpy as np
import TrialSlicer
import OpenElectrophy as OE

ALL_CHANNELS_FILENAME = 'NEURAL_CHANNELS_TO_GET'
GROUPED_CHANNELS_FILENAME = 'NEURAL_CHANNEL_GROUPINGS'
ANALOG_CHANNELS_FILENAME = 'ANALOG_CHANNELS'
TIMESTAMPS_FILENAME = 'TIMESTAMPS'

def write_channel_numbers(filename, list_of_lists):
    """Writes a list of lists of channel numbers to a file.
    
    Each list in list_of_lists in written to a line of the file.
    Each line contains channel numbers with a single space following each.
    """
    fi = file(filename, 'w')
    for chlist in list_of_lists:
        for ch in chlist: 
            fi.write(str(ch) + ' ')
        fi.write('\n')
    fi.close()

def read_channel_numbers(filename):
    """ Load TETRODE_CHANNELS control file
    TODO: make sure this works for case of 1-trodes
    This is funny-looking because not all tetrodes have same number
    of channels!
    """
    f = file(filename)
    x = f.readlines()
    f.close()
    return [[int(c) for c in r] for r in [str.split(rr) for rr in x]]

class RecordingSession:
    """Object linked to a directory containing data for processing.
    
    Provides methods to read and write data from that directory.
    """
    def __init__(self, dirname):
        """Create object linked to a data directory
        
        If directory does not exist, will create it.
        """
        self.full_path = dirname
        self.session_name = os.path.basename(os.path.dirname(self.full_path))
        
        if not os.path.exists(self.full_path):
            os.mkdir(self.full_path)
    
    def get_db_filename(self):
        return os.path.join(self.full_path, self.session_name + '.db')
    
    def open_db(self):
        OE.open_db('sqlite:///' + self.get_db_filename())
    
    def get_ns5_loader(self):
        """Returns Loader object for ns5 file"""
        l = ns5.Loader(filename=self.get_ns5_filename())
        l.load_file()
        return l
    
    def read_channel_groups(self):
        """Returns a list of channel groups"""
        return read_channel_numbers(\
            os.path.join(self.full_path, GROUPED_CHANNELS_FILENAME))
    
    def read_neural_channel_ids(self):
        """Returns a list of all channel numbers"""
        return read_channel_numbers(\
            os.path.join(self.full_path, ALL_CHANNELS_FILENAME))[0]
    
    def read_analog_channel_ids(self):
        """Returns a list of analog channels, or None if no file"""
        try:
            return read_channel_numbers(\
                os.path.join(self.full_path, ANALOG_CHANNELS_FILENAME))[0]
        except IOError:
            return None

    def write_channel_groups(self, list_of_lists):
        """Writes metadata for channel groupings."""
        write_channel_numbers(\
            os.path.join(self.full_path, GROUPED_CHANNELS_FILENAME),
            list_of_lists)
    
    def write_neural_channel_ids(self, list_of_channels):
        """Writes list of channels to be processed."""
        write_channel_numbers(\
            os.path.join(self.full_path, ALL_CHANNELS_FILENAME),
            [list_of_channels])
    
    def write_analog_channel_ids(self, list_of_channels):
        """Writes analog channel numbers."""
        write_channel_numbers(\
            os.path.join(self.full_path, ANALOG_CHANNELS_FILENAME),
            [list_of_channels])
    
    def get_ns5_filename(self):
        """Returns ns5 filename in recording session"""
        filename_list = glob.glob(os.path.join(self.full_path, '*.ns5'))
        if len(filename_list) == 0:
            return None        
        
        if len(filename_list) > 1:
            print "warning: multiple ns5 files exist in %s" % self.full_path
        
        return filename_list[0]
    
    def add_file(self, filename):
        """Copy file into RecordingSession."""
        shutil.copy(filename, self.full_path)
    
    def add_timestamps(self, list_of_values):
        """Adds timestamps by writing values to file in directory"""
        # different format, one value per line
        list_to_write = [[v] for v in list_of_values]        
        write_channel_numbers(os.path.join(self.full_path, TIMESTAMPS_FILENAME),
            list_to_write)
    
    def read_timestamps(self):
        t = read_channel_numbers(\
            os.path.join(self.full_path, TIMESTAMPS_FILENAME))
        return np.array([tt[0] for tt in t])
    
    def put_neural_data_into_db(self, soft_limits_sec=(-2., 2.25), 
        hard_limits_sec=(-.25, .5)):
        """Loads neural data from ns5 file and puts into OE database.
        
        Slices around times provided in TIMESTAMPS.
        """
        OE.open_db(url='sqlite:///' + self.get_db_filename())
        session = OE.Session()
        
        t = self.read_timestamps()
        l = self.get_ns5_loader()
        hard_limits = np.array(\
            np.asarray(hard_limits_sec) * l.header.f_samp, dtype=np.int)
        soft_limits = np.array(\
            np.asarray(soft_limits_sec) * l.header.f_samp, dtype=np.int)
        
        t_starts, t_stops = TrialSlicer.slice_trials(\
            timestamps=t,
            soft_limits=soft_limits, 
            hard_limits=hard_limits, 
            meth='end_of_previous', 
            data_range=(0, l.header.n_samples))
        
        blr = OE.neo.io.BlackrockIO(self.get_ns5_filename())
        
        block = blr.read_block(full_range=8192., t_starts=t_starts, 
            t_stops=t_stops, chlist=self.read_neural_channel_ids())
        
        
        block2 = OE.io.io.hierachicalNeoToOe(block)
        block2.save(session=session)
        
        return block

