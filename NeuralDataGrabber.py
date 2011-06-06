import sys
import os
import shutil
import time
import numpy as np
import glob

class NeuralDataGrabber:
    """Finds *.ns5 files and builds a data analysis directory."""
    def __init__(self, filename_prefix, ndaq_dir='/media/alfshr/REPO_NDAQ',
        data_analysis_dir='.', 
        bcontrol_data_dir=None,
        shove_channels=None, tetrode_channels=None):
        """Builds a new object to get data from a certain location.
        
        This object needs to know where the raw *.ns5 files are located.
        It makes some assumptions about the filenameing scheme. Given a
        date, it extracts files from that date and links them to a
        well-formed directory tree in `data_analysis_dir`. It will also
        grab the other necessary files.
        
        Parameters
        ----------
        filename_prefix : string, the beginning of the *.ns5 filename to
            acquire. Typical value would be 'datafile_CR_CR12B_'
        ndaq_dir : directory where *.ns5 files are located
        data_analysis_dir : root of tree to place files
        bcontrol_data_dir : directory to find behaving *.mat files
        shove_channels : filename to SHOVE_CHANNELS, or None
        tetrode_channels : filename to TETRODE_CHANNELS, or None
        
        Perhaps this should accept only ratname and build the rest.
        Actually perhaps another object should know how to build the filename
        from the bits and pieces.
        """
        self.filename_prefix = filename_prefix
        self.ndaq_dir = ndaq_dir
        self.data_analysis_dir = data_analysis_dir
        self.bcontrol_data_dir = bcontrol_data_dir
        self.shove_channels = shove_channels
        self.tetrode_channels = tetrode_channels
        
        if self.data_analysis_dir is not None:
            if not os.path.exists(self.data_analysis_dir):
                os.mkdir(self.data_analysis_dir)
    
    def get_date(self, date_string, shove_channels=None, tetrode_channels=None,
        verbose=False):
        """Finds files from certain date and builds data analysis directory.
        
        Typical value for date_string : '110513'
        Will try to get all files from that date (perhaps this should
        be overrideable).
        Will assume that '001' is the behaving file and try to find the matfile.
        
        If `shove_channels` and/or `tetrode_channels` is specified, these
        will be used instead of object defaults.
        """
        if shove_channels is None:
            shove_channels = self.shove_channels
        elif not os.path.exists(shove_channels):
            print "warning: %s does not exist for shove_channels" % shove_channels
        if tetrode_channels is None:
            tetrode_channels = self.tetrode_channels
        elif not os.path.exists(tetrode_channels):
            print "warning: %s does not exist for tetrode_channels" % tetrode_channels

        # Make a directory for this date
        date_dir = os.path.join(self.data_analysis_dir, date_string)
        if not os.path.exists(date_dir):
            os.mkdir(date_dir)
        
        # Handle 001 and 002 files, if they exist
        if os.path.exists(os.path.join(self.ndaq_dir, 
            self.filename_prefix  + date_string + '_001.ns5')):
            self.handle_behaving(date_dir, date_string, shove_channels, 
                tetrode_channels, verbose)
        else:
            print ("warning: no 001 file found for %s" % date_string)

        if os.path.exists(os.path.join(self.ndaq_dir, 
            self.filename_prefix  + date_string + '_002.ns5')):
            self.handle_WN(date_dir, date_string, shove_channels, tetrode_channels)
        else:
            print ("warning: no 002 file found for %s" % date_string)
    
    def handle_behaving(self, date_dir, date_string, shove_channels, 
        tetrode_channels, verbose=False, force_run=False):
        # Create a directory for behaving files
        final_dir = os.path.join(date_dir, '001')        
        if not os.path.exists(final_dir):
            os.mkdir(final_dir)
        
        # Follow usual protocol for creating behaving filename
        ndata_filename = os.path.join(self.ndaq_dir,
            self.filename_prefix + date_string + '_001.ns5')        
        
        # Actually link it
        self.link_file(ndata_filename, final_dir, verbose=verbose, 
            force_run=force_run)
        
        # Copy the meta files
        if shove_channels is not None and os.path.exists(shove_channels):
            shutil.copyfile(shove_channels, 
                os.path.join(final_dir, 'SHOVE_CHANNELS'))
        if tetrode_channels is not None and os.path.exists(tetrode_channels):
            shutil.copyfile(tetrode_channels, 
                os.path.join(final_dir, 'TETRODE_CHANNELS'))
        
        # Deal with *.mat file here
        if self.bcontrol_data_dir is not None:
            # Only run if force_run or if no bdata mat file exists
            if force_run or \
                (len(glob.glob(os.path.join(final_dir, 'data_*.mat'))) == 0):
                # Get closest file
                ns5_file_time = os.path.getmtime(ndata_filename)
                bdata_filename = self.choose_bdata_file(ns5_file_time, verbose)
                bdata_filename = os.path.join(self.bcontrol_data_dir, 
                    bdata_filename)
                
                # Copy it
                shutil.copy(bdata_filename, final_dir)
            
            # Test if multiple bdata mat files now exist
            if len(glob.glob(os.path.join(final_dir, 'data_*.mat'))) > 1:
                print "warning: bcontrol matfiles already exist."
                print "you should delete extras in %s" % final_dir
            
    
    def choose_bdata_file(self, ns5_file_time, verbose=False, pre_time=-1000., 
        post_time=1000.):
        # List all behavior files in the appropriate directory
        bc_files = os.listdir(self.bcontrol_data_dir)
        bc_files.sort()
        bc_files = [os.path.join(self.bcontrol_data_dir, fi) \
            for fi in bc_files]

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
    
    def handle_WN(self, date_dir, date_string, shove_channels, tetrode_channels):
        # Handle WN
        final_dir = os.path.join(date_dir, '002')        
        if not os.path.exists(final_dir):
            os.mkdir(final_dir)
        
        # Find the 002 file
        file_to_find = os.path.join(self.ndaq_dir,
            self.filename_prefix + date_string + '_002.ns5')
        
        # Actually link it
        self.link_file(file_to_find, final_dir)
        
        # Copy the meta files
        if shove_channels is not None and os.path.exists(shove_channels):
            shutil.copyfile(shove_channels, 
                os.path.join(final_dir, 'SHOVE_CHANNELS'))
        if tetrode_channels is not None and os.path.exists(tetrode_channels):
            shutil.copyfile(tetrode_channels, 
                os.path.join(final_dir, 'TETRODE_CHANNELS'))
    
    def link_file(self, filename, final_dir, verbose=False, 
        dryrun=False, force_run=False):
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