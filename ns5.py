import numpy as np
import struct
    

class HeaderInfo:
    """Holds information from the ns5 file header about the file.
    
    """
    pass    
    

class Loader(object):
    def __init__(self, settings=None, filename=None):
        """Specifies settings for the new loader (channel mapping, etc).
        
        
        Parameters
        ----------
        settings : string
            Currently, the only supported string is 'HO4'. If this string
            is used, the correct channel mappings for this type of data
            will be loaded.
        
        
        If settings is None, then you can still access channels by number
        using get_channel and get_channel_as_array
        
        Audio channels are available as get_analog_channel
        """
        self.filename = filename
        
        #~ if settings is None:
            #~ print "No settings specified, using default value: HO4"
            #~ settings = 'HO4'
        
        
        if settings == 'HO4':
            self.KEEP_CHANNELS = \
                [16, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27, 30, 29, 32]
        
        self._mm = None
        self.file_handle = None


    def load_file(self, filename=None):
        """Loads an ns5 file, if not already done. Stores in hdf5 file.
        
        
        Parameters
        ----------
        filename : string
            Location of the *.ns5 binary file to be loaded. This file
            should contain both neural data and the audio waveforms
            that were played.
        
        
        Outputs
        -------       
        self.neural_data : np.memmap (TODO: MinMemMap)
            A handle to allow access to the hdf5 neural data
        
        self.audio_data : np.memmap (TODO: MinMemMap)
            A handle to allow access to the hdf5 audio data
        
        self.header : HeaderInfo

        
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


    def _parse_channels(self):
        """Identifies desired neural and audio channels in the data."""
        # determine the channels to keep from the raw data file
        self._idxs_into_memory_map = \
            [self.header.Channel_ID.index(nc) \
            if self.header.Channel_ID.__contains__(nc) \
            else None for nc in self.KEEP_CHANNELS]
        if None in self._idxs_into_memory_map:
            bad_channel = self.KEEP_CHANNELS[map(\
                bool, self._idxs_into_memory_map).index(False)]
            print "You requested channel %d that does not exist \
                in the datafile." % bad_channel
            self._idxs_into_memory_map = filter(bool, idxs_into_memory_map)
        
        # determine the audio channels
        self._audio_idxs_into_memory_map = \
            [self.header.Channel_ID.index(nc) \
            if self.header.Channel_ID.__contains__(nc) \
            else None for nc in [135, 136] ]
        if None in self._audio_idxs_into_memory_map:            
            print "Missing audio in neural file."
            self._audio_idxs_into_memory_map = filter(bool, 
                self._audio_idxs_into_memory_map)

    
    def regenerate_memmap(self):
        """user calls me to reclaim memory
        
        slices of _mm are doled out intelligently
        i am not sure if those slices need to be deleted by the user?
        i am guessing that they will be broken after calling regenerate
        at worst, not deleting them may break the regeneration of _mm
        
        test this
        """
        try:
            del self._mm
        except AttributeError: 
            pass
        
        self._mm = np.memmap(\
            self.filename, dtype='h', mode='r', 
            offset=self.header.Header, 
            shape=(self.header.n_samples, self.header.Channel_Count))
    
    
    #~ def get_neural_channel(self, channel_id):
        #~ if channel_id in self.KEEP_CHANNELS:
            #~ mm_index = self.KEEP_CHANNELS.index(channel_id)
            
        #~ else:
            #~ return None
        
    
    def get_neural_memmap(self):
        """Returns a memmap to just the neural channels in keep_channels"""
        self._parse_channels()
        
        if '_mm' not in self.__dict__: self.regenerate_memmap()
        
        
        # I see the problem, this rearrangement of columns cause the load
        # from disk to happen
        return self._mm[:, self._idxs_into_memory_map]

    
    
    def get_audio_memmap(self):        
        """Return a memmap to just the audio channels"""
        self._parse_channels()
        
        if '_mm' not in self.__dict__: self.regenerate_memmap()
        
        return self._mm[:, self._audio_idxs_into_memory_map]        
    
    def __del__(self):
        # try catch doesn't work!
        
        # this deletion doesn't free memory, even though del l._mm does!
        if '_mm' in self.__dict__: del self._mm
        #else: print "gracefully skipping"
        
        #~ try: 
            #~ del self._mm
        #~ except AttributeError: 
            #~ pass
        #~ #super(Loader, self).__del__() #TypeError
    
    def get_channel(self, channel_number):
        # Accept ns5-like channel numbers, translate to indices into
        # _mm via keep_channels, then return
        # Then get_neural and get_audio can call this
        # And _parse_channels will be unnecesaary
        try:
            mm_index = self.header.Channel_ID.index(channel_number)
        except ValueError:
            print "Channel number %d does not exist" % channel_number
            return np.array([])
        
        self.regenerate_memmap()
        return self._mm[:, mm_index]
    
    def get_channel_as_array(self, channel_number):
        """Loads data from memmap, deletes memmap, returns data"""
        
        data = np.array(self.get_channel(channel_number))
        self.regenerate_memmap()
        return data

    def get_analog_channel_as_array(self, analog_chn):
        """Loads data from analog channels, such as audio.
        
        Simply adds 128 to the channel number to convert.
        This is just the way Cyberkinetics numbers its channels.
        """
        
        return self.get_channel_as_array(analog_chn + 128)
    
    def get_all_analog_channels(self):
        """Returns all analog channels.
        
        That is, those with channel IDs >128 and <=144.
        """
        # Detect audio channels
        audio_channels = self.get_audio_channel_numbers() - 128
        
        # Grab audio data, may be mono or stereo
        return np.array([self.get_analog_channel_as_array(ch) \
            for ch in audio_channels])        

    def get_audio_channel_numbers(self):
        """Returns index of audio channels."""
        # Detect audio channels
        channel_array = np.array(self.header.Channel_ID)
        audio_channels = channel_array[np.nonzero(\
            (channel_array > 128) & \
            (channel_array <= 144))]
        return audio_channels

