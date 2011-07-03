"""Module containing experiment-specific code to create RecordingSessions.
* creates proper RecordingSession by copying necessary files

Experimental-specific
* ns5 filenaming conventions
* logic to find bcontrol behavior files

Wraps RecordingSession
* loads audio data, detects onsets, tells RecordingSession the timestamps
"""

import os
import time
import RecordingSession
import numpy as np

# Functions to parse the way I name my ns5 files
def make_filename(dirname=None, username='*', ratname='*', date='*', number='*'):
    """Creates an ns5 filename from its constituent parts."""
    filename = '_'.join(['datafile', username, ratname, date, number]) + '.ns5'
    if dirname is None:
        return filename
    else:
        return os.path.join(dirname, filename)

def get_field(filename, fieldname):
    """Get a field from an ns5 filename.
    
    Options are 'username', 'ratname', 'date', 'number'.
    """
    d = {'username': 1, 'ratname': 2, 'date': 3, 'number': 4}
    idx = d[fieldname]
    
    f = os.path.split(filename)[1]
    f = os.path.splitext(f)[0]
    
    components = f.split('_')
    return components[idx]

# Function to link a file
# Could probably go in RecordingSession but it's a little ugly because
# uses system call.
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

# Many of the following functions work like wrapper for RecordingSession.
# Function to load audio data from raw file and detect onsets
def add_timestamps_to_session(rs, manual_threshhold=None, minimum_duration_ms=50,
    pre_first=0, post_last=0):
    """Given a RecordingSession, makes TIMESTAMPS file.
    
    I get the right channels and filenames from RecordingSession, then call
    `calculate_timestamps`, then tell the session what I've calculated.
    """
    # Get data from recording session
    filename = rs.get_ns5_filename()
    try:
        audio_channels = rs.read_analog_channels()
    except IOError:
        # Recording session doesn't know what analog channels are supposed to be
        audio_channels = None
    
    # Calculate timestamps
    t = calculate_timestamps(filename, manual_threshhold, audio_channels,
        minimum_duration_ms, pre_first, post_last)
    
    # Tell RecordingSession about them
    rs.add_timestamps(t)

def calculate_timestamps(filename, manual_threshhold=None, audio_channels=None, 
    minimum_duration_ms=50, pre_first=0, post_last=0)
    """Given ns5 file and channel numbers, returns audio onsets.
    
    I uses `ns5.Loader` to open ns5 file and `AudioTools.OnsetDetector`
    to identify onsets. I discard onsets too close to the ends.
    
    Input
    -----
    filename : path to *.ns5 binary file
    manual_threshhold, minimum_duration_ms : Passed to OnsetDetector    
    audio_channels : the channel numbers to analyze for onsets. If None,
        I'll use the first channel, or the first two channels if available.    
    pre_first : audio data before this sample will be ignored; therefore,
        the first onset will be at least this.
    post_last : audio data after this sample will be ignored; therefore,
        the last onset will be at most this.
    
    Returns
    ------
    Array of detected stimulus onsets in samples
    """
    # Load ns5 file
    l = ns5.Loader(filename=filename)
    l.load_file()
    
    # Grab audio data, may be mono or stereo
    if audio_channels is None:
        audio_data = l.get_all_analog_channels()
    else:
        # Specify channels manually
        audio_data = np.array(\
            [l.get_analog_channel_as_array(ch) for ch in audio_channels])
    
    # Plot the data, there may be artefacts to be truncated
    if debug_mode:
        import matplotlib.pyplot as plt
        plt.plot(audio_data.transpose())
        plt.show()
        1/0

    # Truncate artefacts
    if truncate_data is not None:
        audio_data = audio_data[:,:truncate_data]

    # Onset detector expect mono to be 1d, not Nx1
    if len(audio_channels) == 1:
        # Mono
        audio_data = audio_data.flatten()

    # Instantiate an onset detector
    od = AudioTools.OnsetDetector(input_data=audio_data,
        F_SAMP=l.header.f_samp,
        plot_debugging_figures=True, 
        manual_threshhold=manual_threshhold,
        minimum_duration_ms=minimum_duration_ms)
    
    # Run it and save output to disk
    od.execute()
    
    do = od.detected_onsets
    if drop_first_and_last:
        do = do[1:-1]
    np.savetxt(os.path.join(os.path.split(filename)[0], 'TIMESTAMPS'), do,
        fmt='%d')

    return do


# Functions to find and add bcontrol data to RecordingSession
# Acts like RecordingSession wrapper.
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



# Utility function using experimental logic to find the right behavior file.
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
