"""Helper functions and variables for all data in a recording session.

This allows you to create a well-formed directory for ns5_process and fill
it with data.

Try to keep logic relating to any specific experiment out of this module.
"""

import shutil
import glob
import os.path

ALL_CHANNELS_FILENAME = 'SHOVE_CHANNELS'
GROUPED_CHANNELS_FILENAME = 'TETRODE_CHANNELS'
ANALOG_CHANNELS_FILENAME = 'ANALOG_CHANNELS'

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
    
    """
    def __init__(self, root_directory, name):        
        self.name = name
        self.root_directory = root_directory
        self.full_path = os.path.join(self.root_directory, self.name)
        
        if not os.path.exists(self.root_directory):
            os.mkdir(self.root_directory)
        
        if not os.path.exists(self.full_path):
            os.mkdir(self.full_path)
    
    def read_channel_groups(self):
        """Returns a list of channel groups"""
        return read_channel_numbers(GROUPED_CHANNELS_FILENAME)
    
    def read_all_channels(self):
        """Returns a list of all channel numbers"""
        return read_channel_numbers(ALL_CHANNELS_FILENAME)[0]
    
    def read_analog_channels(self):
        """Returns a list of analog channels"""
        return read_channel_numbers(ANALOG_CHANNELS_FILENAME)[0]

    def write_channel_groups(self, list_of_lists):
        """Writes metadata for channel groupings."""
        write_channel_numbers(\
            os.path.join(self.full_path, GROUPED_CHANNELS_FILENAME),
            list_of_lists)
    
    def write_all_channels(self, list_of_channels):
        """Writes list of channels to be processed."""
        write_channel_numbers(\
            os.path.join(self.full_path, ALL_CHANNELS_FILENAME),
            [list_of_channels])
    
    def write_analog_channels(self, list_of_channels):
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


