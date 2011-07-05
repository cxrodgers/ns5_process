import numpy as np
import struct
    
class HeaderInfo:
    """Holds information from the ns5 file header about the file."""
    pass       

class Loader(object):
    """Object to load data from binary ns5 files.
    
    Methods
    -------
    load_file : actually create links to file on disk
    load_header : load header info and store in self.header
    get_channel_as_array : Returns 1d numpy array of the entire recording
        from requested channel.
    get_analog_channel_as_array : Same as get_channel_as_array, but works
        on analog channels rather than neural channels.
    get_analog_channel_ids : Returns an array of analog channel numbers
        existing in the file.    
    get_neural_channel_ids : Returns an array of neural channel numbers
        existing in the file.
    regenerate_memmap : Deletes and restores the underlying memmap, which
        may free up memory.
    
    Issues
    ------
    Memory leaks may exist
    Not sure that regenerate_memmap actually frees up any memory.
    """
    def __init__(self, filename=None):
        """Creates a new object to load data from the ns5 file you specify.

        filename : path to ns5 file
        Call load_file() to actually get data from the file.
        """
        self.filename = filename
        
        
        
        self._mm = None
        self.file_handle = None

    def load_file(self, filename=None):
        """Loads an ns5 file, if not already done.
        
        *.ns5 BINARY FILE FORMAT
        The following information is contained in the first part of the header
        file.
        The size in bytes, the variable name, the data type, and the meaning are
        given below. Everything is little-endian.

        8B. File_Type_ID. char. Always "NEURALSG"
        16B. File_Spec. char. Always "30 kS/s\0"
        4B. Period. uint32. Always 1.
        4B. Channel_Count. uint32. Generally 32 or 34.
        Channel_Count*4B. uint32. Channel_ID. One uint32 for each channel.

        Thus the total length of the header is 8+16+4+4+Channel_Count*4.
        Immediately after this header, the raw data begins.
        Each sample is a 2B signed int16.
        For our hardware, the conversion factor is 4096.0 / 2**16 mV/bit.
        The samples for each channel are interleaved, so the first Channel_Count
        samples correspond to the first sample from each channel, in the same
        order as the channel id's in the header.

        Variable names are consistent with the Neuroshare specification.
        """
        # If filename specified, use it, else use previously specified
        if filename is not None: self.filename = filename
        
        # Load header info into self.header
        self.load_header()
        
        # build an internal memmap linking to the data on disk
        self.regenerate_memmap()
    
    def load_header(self, filename=None):
        """Reads ns5 file header and writes info to self.header"""
        # (Re-)initialize header
        self.header = HeaderInfo()
        
        # the width of each sample is always 2 bytes
        self.header.sample_width = 2
        
        # If filename specified, use it, else use previously specified
        if filename is not None: self.filename = filename
        self.header.filename = self.filename
        
        # first load the binary in directly
        self.file_handle = open(self.filename, 'rb') # buffering=?

        # Read File_Type_ID and check compatibility
        # If v2.2 is used, this value will be 'NEURALCD', which uses a slightly
        # more complex header. Currently unsupported.
        self.header.File_Type_ID = [chr(ord(c)) \
            for c in self.file_handle.read(8)]
        if "".join(self.header.File_Type_ID) != 'NEURALSG':
            print "Incompatible ns5 file format. Only v2.1 is supported.\n" + \
                "This will probably not work."          
        
        
        # Read File_Spec and check compatibility.
        self.header.File_Spec = [chr(ord(c)) \
            for c in self.file_handle.read(16)]
        if "".join(self.header.File_Spec[:8]) != '30 kS/s\0':
            print "File_Spec seems to indicate you did not sample at 30KHz."
        
        
        #R ead Period and verify that 30KHz was used. If not, the code will
        # still run but it's unlikely the data will be useful.
        self.header.period, = struct.unpack('<I', self.file_handle.read(4))
        if self.header.period != 1:
            print "Period seems to indicate you did not sample at 30KHz."
        self.header.f_samp = self.header.period * 30000.0


        # Read Channel_Count and Channel_ID
        self.header.Channel_Count, = struct.unpack('<I',
            self.file_handle.read(4))
        self.header.Channel_ID = [struct.unpack('<I',
            self.file_handle.read(4))[0]
            for n in xrange(self.header.Channel_Count)]
        
        # Compute total header length
        self.header.Header = 8 + 16 + 4 + 4 + \
            4*self.header.Channel_Count # in bytes

        # determine length of file
        self.file_handle.seek(0, 2) # last byte
        self.header.file_total_size = self.file_handle.tell()
        self.header.n_samples = \
            (self.header.file_total_size - self.header.Header) / \
            self.header.Channel_Count / self.header.sample_width
        self.header.Length = np.float64(self.header.n_samples) / \
            self.header.Channel_Count
        if self.header.sample_width * self.header.Channel_Count * \
            self.header.n_samples + \
            self.header.Header != self.header.file_total_size:
            print "I got header of %dB, %d channels, %d samples, \
                but total file size of %dB" % (self.header.Header, 
                self.header.Channel_Count, self.header.n_samples, 
                self.header.file_total_size)

        # close file
        self.file_handle.close()

    
    def regenerate_memmap(self):
        """Delete internal memmap and create a new one, to save memory."""
        try:
            del self._mm
        except AttributeError: 
            pass
        
        self._mm = np.memmap(\
            self.filename, dtype='h', mode='r', 
            offset=self.header.Header, 
            shape=(self.header.n_samples, self.header.Channel_Count))
    
    def __del__(self):
        # this deletion doesn't free memory, even though del l._mm does!
        if '_mm' in self.__dict__: del self._mm
        #else: print "gracefully skipping"
    
    def _get_channel(self, channel_number):
        """Returns slice into internal memmap for requested channel"""
        try:
            mm_index = self.header.Channel_ID.index(channel_number)
        except ValueError:
            print "Channel number %d does not exist" % channel_number
            return np.array([])
        
        self.regenerate_memmap()
        return self._mm[:, mm_index]
    
    def get_channel_as_array(self, channel_number):
        """Returns data from requested channel as a 1d numpy array."""
        data = np.array(self._get_channel(channel_number))
        self.regenerate_memmap()
        return data

    def get_analog_channel_as_array(self, analog_chn):
        """Returns data from requested analog channel as a numpy array.
        
        Simply adds 128 to the channel number to convert to ns5 number.
        This is just the way Cyberkinetics numbers its channels.
        """
        return self.get_channel_as_array(analog_chn + 128)    

    def get_audio_channel_numbers(self):
        """Deprecated, use get_analog_channel_ids"""
        return self.get_analog_channel_ids()
    
    def get_analog_channel_ids(self):
        """Returns array of analog channel ids existing in the file.
        
        These can then be loaded by calling get_analog_channel_as_array(chn).
        """
        return np.array(filter(lambda x: (x > 128) and (x <= 144), 
            self.header.Channel_ID)) - 128

    def get_neural_channel_numbers(self):
        return np.array(filter(lambda x: x <= 128, self.header.Channel_ID))

