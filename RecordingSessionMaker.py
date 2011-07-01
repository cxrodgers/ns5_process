"""Module containing experiment-specific code to create RecordingSessions"""

import os
import time
import RecordingSession
import numpy as np


def link_file(filename, final_dir, verbose=False, dryrun=False, force_run=True):
    """Creates a link from source `filename` to target directory.
    
    final_dir : target directory, must exist
    verbose : if True, prints command issued
    dryrun : if True, do nothing
    force_run : if True, overwrite link if it exists
    
    The underlying call is to `ln` so this must be available on your
    system.
    """
    # Error check existence
    if not os.path.exists(filename):
        raise IOError("Can't find file %s" % filename)
    if not os.path.exists(final_dir):
        raise IOError("Target dir %s does not exist" % final_dir)
    
    # Link location
    target_filename = os.path.join(final_dir, os.path.split(filename)[1])
    if os.path.exists(target_filename) and not force_run:
        if verbose:
            print "link already exists, giving up"
        return
    if os.path.exists(target_filename) and force_run:
        if verbose:
            print "link already exists, deleting"
        if not dryrun:
            os.remove(target_filename)
    
    # Do the link
    sys_call_str = 'ln -s %s %s' % (filename, target_filename)
    if verbose:
        print sys_call_str
    if not dryrun:
        os.system(sys_call_str)


class RecordingSessionMaker:
    """Create a RecordingSession and add data to it.
    
    This class handles experiment-specific crap, like linking the data
    file, subtracting broken channels, finding and adding in bcontrol files.
    """   
    def __init__(self, data_analysis_dir, all_channels_file,
        channel_groups_file):
        """Builds a new object to get data from a certain location.
        
        Given the data location and the channel numbering, I can produce
        a well-formed data directory for each recording session to analyze.
        Each recording session can specify the type of recording
        (behaving, passive) and broken channels on that day.
        
        You probably will want to create a new object for each subject, since
        the channel numberings and data locations may differ.
        
        Parameters
        ----------
        data_analysis_dir : root of tree to place files (will be created)
        all_channels_file : filename containing list of channel numbers
        channel_groups_file : filename containing channel groups
        
        Both of the channel files should follow the format that
        RecordingSession expects.
        """
        # Store input directory and create if necessary
        self.data_analysis_dir = data_analysis_dir
        if not os.path.exists(self.data_analysis_dir):
            os.mkdir(self.data_analysis_dir)
        
        # Load channel numberings from disk. SHOVE_CHANNELS should be just
        # one line long.
        self.SC_list = \
            RecordingSession.read_channel_numbers(all_channels_file)[0]
        self.TC_list = \
            RecordingSession.read_channel_numbers(channel_groups_file)
    
    def make_session(self, ns5_filename, session_name, remove_from_SC=[], 
        remove_from_TC=[]):
        """Create a RecordingSession for some data.
        
        ns5_filename : full path to raw ns5 file
        session_name : name of subdirectory, eg '110413_LBPB'
        remove_from_SC : channels to remove from SHOVE_CHANNELS
        remove_from_TC : channels to remove from TETRODE_CHANNELS

        Returns the created RecordingSession.
        """
        # Create RecordingSession
        rs = RecordingSession.RecordingSession(self.data_analysis_dir,
            session_name)
        
        # Link neural data to session dir, overwriting if necessary
        link_file(ns5_filename, rs.full_path)
        
        # Create channel numbering meta file
        session_SC_list = sorted(list(set(self.SC_list) - set(remove_from_SC)))
        rs.write_all_channels(session_SC_list)
        
        # Create channel grouping meta file
        session_TC_list = [sorted(list(set(this_ch) - set(remove_from_TC)))\
            for this_ch in self.TC_list]
        rs.write_channel_groups(session_TC_list)
    
        return rs


    
def add_bcontrol_data_to_session(bcontrol_folder, session, verbose=False, 
    pre_time=-1000., post_time=1000.):
    """Copies the best bcontrol file into the session

    For specifics, see `find_closest_bcontrol_file`.
    """
    # Get time of file    
    ns5_filename = session.get_ns5_filename()
    
    # Find best file
    bdata_filename = find_closest_bcontrol_file(bcontrol_folder, 
        ns5_filename, verbose, pre_time, post_time)
    
    # Add to session
    session.add_file(bdata_filename)

def find_closest_bcontrol_file(bcontrol_folder, ns5_filename, verbose=False,
    pre_time=-1000., post_time=1000.):
    """Returns the best matched bcontrol file for a RecordingSession.
    
    Chooses the bcontrol mat-file in the directory that is the latest
    within the specified window. If no files are in the window, return the
    closest file and issue warning.
    
    bcontrol_folder : directory containing bcontrol files\
    session : RecordingSession to add behavioral data to
    verbose : print debugging info
    pre_time : negative, time before ns5 file time
    post_time : positive, time after ns5 file time
    """
    
    # Get file time
    ns5_file_time = os.path.getmtime(ns5_filename)
    
    # List all behavior files in the appropriate directory
    bc_files = os.listdir(bcontrol_folder)
    bc_files.sort()
    bc_files = [os.path.join(bcontrol_folder, fi) for fi in bc_files]

    # Get file times of all files in bcontrol data. Use time.ctime()
    # to convert to human-readable.
    file_times = \
        np.array([os.path.getmtime(bc_file) for bc_file in bc_files])
    
    # Find files in the directory that have times within the window
    potential_file_idxs = np.where(
        (np.abs(file_times - ns5_file_time) < post_time) &
        (np.abs(file_times - ns5_file_time) > pre_time))
    
    # Output debugging information
    if verbose:
        print "BEGIN FINDING FILE"
        print "target file time: %f = %s" % \
            (ns5_file_time, time.ctime(ns5_file_time))
        print "found %d potential matches" % len(potential_file_idxs[0])
        for pfi in potential_file_idxs[0]:
            print "%s : %f" % (time.ctime(file_times[pfi]),
                file_times[pfi] - ns5_file_time)

    # Choose the last file in the range
    if len(potential_file_idxs[0]) > 0:
        idx = potential_file_idxs[0][-1]
    else:
        # Nothing within range, use closest file
        idx = np.argmin(np.abs(file_times - ns5_file_time))
        print "warning: no files within target range (" + \
            time.ctime(ns5_file_time) + "), using " +  \
            time.ctime(os.path.getmtime(bc_files[idx]))
    found_file = bc_files[idx]
    
    # Print debugging information
    if verbose:
        print "Found file at time %s and the diff is %f" % (found_file, 
            os.path.getmtime(found_file) - ns5_file_time)

    # Return path to best file choice
    return found_file
