import sys
import os
import shutil

class NeuralDataGrabber:
    """Finds *.ns5 files and builds a data analysis directory."""
    def __init__(self, filename_prefix, ndaq_dir='/media/alfshr/REPO_NDAQ',
        data_analysis_dir='.', shove_channels=None, tetrode_channels=None):
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
        shove_channels : filename to SHOVE_CHANNELS, or None
        tetrode_channels : filename to TETRODE_CHANNELS, or None
        
        Perhaps this should accept only ratname and build the rest.
        Actually perhaps another object should know how to build the filename
        from the bits and pieces.
        """
        self.filename_prefix = filename_prefix
        self.ndaq_dir = ndaq_dir
        self.data_analysis_dir = data_analysis_dir
    
    def get_date(self, date_string, shove_channels=None, tetrode_channels=None):
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
        if tetrode_channels is None:
            tetrode_channels = self.tetrode_channels
        
        # Make a directory for this date
        date_dir = os.path.join(self.data_analysis_dir, date_string)
        final_dir = os.path.join(date_dir, '001')
        if not os.path.exists(date_dir):
            os.mkdir(date_dir)
        if not os.path.exists(final_dir):
            os.mkdir(final_dir)
        
        # Find the 001 file
        file_to_find = os.path.join(self.ndaq_dir,
            self.filename_prefix + date_string + '_001.ns5')        
        
        # Actually link it
        self.get_behaving_file(file_to_find, final_dir)
        
        # Copy the meta files
        if shove_channels is not None and os.path.exists(shove_channels):
            shutil.copyfile(shove_channels, 
                os.path.join(final_dir, 'SHOVE_CHANNELS'))
        if tetrode_channels is not None and os.path.exists(tetrode_channels):
            shutil.copyfile(tetrode_channels, 
                os.path.join(final_dir, 'TETRODE_CHANNELS'))
        
    
    def get_behaving_file(self, filename, final_dir, verbose=True, 
        dryrun=False, force_run=False):
        if not os.path.exists(filename):
            raise IOError("Can't find file %s" % filename)
        if not os.path.exists(final_dir):
            raise IOError("Target dir %s does not exist" % final_dir)
        
        # Link location
        target_filename = os.path.join(final_dir, os.path.split(filename)[1])
        if os.path.exists(target_filename) and not force_run:
            print "link already exists, giving up"
            return
        if os.path.exists(target_filename) and force_run:
            print "link already exists, deleting"
            if not dryrun:
                os.remove(target_filename)
        
        # Do the link
        sys_call_str = 'ln -s %s %s' % (filename, target_filename)
        if verbose:
            print sys_call_str
        if not dryrun:
            os.system(sys_call_str)