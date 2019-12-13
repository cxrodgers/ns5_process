from __future__ import print_function
from __future__ import division
from builtins import chr
from builtins import range
from past.utils import old_div
from builtins import object
import numpy as np
import struct
    
class HeaderInfo(object):
    """Holds information from the ns5 file header about the file."""
    pass       

class Loader(object):
    """Object to load data from binary ns5 files.
    
    This object tries to:
    1) Load data as quickly as possible
    2) Not store any temporary variables
    3) Allow easy access by Cyberkinetics channel number
    
    The problem is that the straightforward solutions to this problem have
    some unintended side effects. As far as I can tell, you cannot specify
    a stride when reading data from disk. Since the data is stored by sample
    first, and then channel, this means that you have to load all the channels
    at once for any given block of time. That means that reading one channel
    is no faster than reading all of them.
    
    Memmaps allow a convenient access routine that possibly sidesteps this
    issue, but I cannot verify that they are actually any faster, and in
    my experience they create memory leaks. So no memmaps are used here.
    
    The other issue is indexing the channel(s) that you want from the 2d block
    of data. I wanted to provide accessor methods that would return one
    channel at a time but there's no way to do this without caching the
    block or re-reading the data.

    To that end the recommended method to use for all operations is
    get_chunk_by_channel. This returns a dict {Cyberkinetics channel number:
    1d array of data}. All channels are read. The dict values are views onto
    the underlying blocks.

    You can access one channel from this dict or iterate over all of them,
    knowing that the expensive read operation only occurs once.

    For backwards compatibility some other methods are provided that index
    this dict for you. Note that you should **not** call these methods
    repeatedly or you will re-read the entire dataset every time. For those
    types of operations, iterate over the dict.
    
    All methods return the data as the underlying datatype (int16).

    Methods
    ---
    get_chunk_by_channel : returns dict {channel : 1d array}
    get_chunk : returns 2d array of data (n_samples, n_channels). For
        the identity of each channel, see self.header.Channel_ID
    get_channel_as_array : Returns one channel from the dict
    get_analog_channel_as_array : Same as get_channel_as_array, but works
        on analog channels rather than neural channels.
    get_analog_channel_ids : Returns an array of analog channel numbers
        existing in the file.    
    get_neural_channel_ids : Returns an array of neural channel numbers
        existing in the file.
    """
    def __init__(self, filename=None):
        """Creates a new object to load data from the ns5 file you specify.

        filename : path to ns5 file
        Call load_file() to actually get data from the file.
        """
        self.filename = filename
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
            print("Incompatible ns5 file format. Only v2.1 is supported.\n" + \
                "This will probably not work.")          
        
        
        # Read File_Spec and check compatibility.
        self.header.File_Spec = [chr(ord(c)) \
            for c in self.file_handle.read(16)]
        if "".join(self.header.File_Spec[:8]) != '30 kS/s\0':
            print("File_Spec seems to indicate you did not sample at 30KHz.")
        
        
        #R ead Period and verify that 30KHz was used. If not, the code will
        # still run but it's unlikely the data will be useful.
        self.header.period, = struct.unpack('<I', self.file_handle.read(4))
        if self.header.period != 1:
            print("Period seems to indicate you did not sample at 30KHz.")
        self.header.f_samp = self.header.period * 30000.0


        # Read Channel_Count and Channel_ID
        self.header.Channel_Count, = struct.unpack('<I',
            self.file_handle.read(4))
        self.header.Channel_ID = [struct.unpack('<I',
            self.file_handle.read(4))[0]
            for n in range(self.header.Channel_Count)]
        
        # Compute total header length
        self.header.Header = 8 + 16 + 4 + 4 + \
            4*self.header.Channel_Count # in bytes

        # determine length of file
        self.file_handle.seek(0, 2) # last byte
        self.header.file_total_size = self.file_handle.tell()
        self.header.n_samples = \
            old_div(old_div((self.header.file_total_size - self.header.Header), \
            self.header.Channel_Count), self.header.sample_width)
        self.header.Length = old_div(np.float64(self.header.n_samples), \
            self.header.Channel_Count)
        if self.header.sample_width * self.header.Channel_Count * \
            self.header.n_samples + \
            self.header.Header != self.header.file_total_size:
            print("I got header of %dB, %d channels, %d samples, \
                but total file size of %dB" % (self.header.Header, 
                self.header.Channel_Count, self.header.n_samples, 
                self.header.file_total_size))

        # close file
        self.file_handle.close()

    
    def get_chunk(self, start=0, n_samples=None, stop=None):
        """Returns a chunk of data with optional start and stop sample.
        
        This is the underlying access method for all other accessors.
        For a nicer interface, see `get_chunk_by_channel`.
        
        As far as I can tell there is no nice way around reading all of the
        channels (without using memmaps, which I want to avoid). 
        
        start : index of first sample.
        stop : like a normal Python stop index, one past the last sample.
            If None, returns data up to the end.
        n_samples : length of each returned channel. (In this case, `stop`
            is ignored.)
        
        Only one of stop or n_samples should be provided.

        Returns 2d numpy array (n_samples, n_channels) with the channels
        in the same order as self.header.Channel_ID.
        """
        # where to start and stop
        start = int(start)
        i1 = self.header.Header + \
            start * self.header.Channel_Count * self.header.sample_width
        if n_samples is None:
            if stop is None:
                stop = self.header.n_samples
            stop = int(stop)
            n_samples = stop - start
            count = n_samples * self.header.Channel_Count
        else:
            n_samples = int(n_samples)
        
        # error check
        if (start + n_samples) > self.header.n_samples:
            print("warning: you have requested more than the available data")
            start = 0
            n_samples = self.header.n_samples
        count = n_samples * self.header.Channel_Count
        
        
        # open file seek to correct place
        fi = file(self.filename)
        fi.seek(i1, 0)
        
        
        # load data and reshape
        res = np.fromfile(file=fi, dtype=np.int16, count=count).reshape(
            (n_samples, self.header.Channel_Count))
        
        return res
    
    def get_chunk_by_channel(self, start=0, n_samples=None, stop=None):
        """Preferred accessor method for all operations.

        start, n_samples, stop : see get_chunk
        
        Returns a dict {Blackrock channel number : 1d array of data}.
        Construction of this dict requires minimal overhead, compared
        to get_chunk, but provides a nicer interface.
        """
        raw = self.get_chunk(start, n_samples, stop)
        res = {}
        
        for n, chn in enumerate(self.header.Channel_ID):
            res[chn] = raw[:, n]
        
        return res
    
    def get_channel(self, channel, start=0, n_samples=None, stop=None):
        if channel not in self.header.Channel_ID:
            raise ValueError("channel %d not in data file" % channel)
        res = self.get_chunk_by_channel(start, n_samples, stop)
        return res[channel]
    
    def get_channel_as_array(self, channel, start=0, n_samples=None, stop=None):
        return self.get_channel(channel, start, n_samples, stop)

    def get_analog_channel_as_array(self, channel, start=0, 
        n_samples=None, stop=None):
        """Returns data from requested analog channel as a numpy array.
        
        Simply adds 128 to the channel number to convert to ns5 number.
        This is just the way Cyberkinetics numbers its channels.
        """
        return self.get_channel(channel + 128, start, n_samples, stop)    

    def get_analog_channel_ids(self):
        """Returns array of analog channel ids existing in the file.
        
        These can then be loaded by calling get_analog_channel_as_array(chn).
        """
        return np.array([x for x in self.header.Channel_ID if (x > 128) and (x <= 144)]) - 128

    def get_neural_channel_ids(self):
        return np.array([x for x in self.header.Channel_ID if x <= 128])

