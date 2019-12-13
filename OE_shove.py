from __future__ import print_function
from __future__ import absolute_import
# The purpose of this script is to load binary data from an ns5 file
# and place the useful part of it into an OE db. The useful part is
# determined by channels and timestamps that are loaded from disk

# Behavior is controlled by the following files which should be in the
# working directory (checked first) or datafile directory.

# SHOVE_CHANNELS: Plaintext list of channel numbers to keep, 1-indexed.
# If SHOVE_CHANNELS does not exist, check for TETRODE_CHANNELS, 1-indexed.
# If TETRODE_CHANNELS does not exist, load all channels.

# TIMESTAMPS: Plaintext list of start times and stop times.
# If TIMESTAMPS contains only a start time, 1sec segments are kept
# If TIMESTAMPS does not exist, the entire file is loaded

# A quantization level of 4096 uV per 32768 bits is assumed.

# The data will be loaded and placed into the db with channel names

# Contains loaders for ns5 files
from . import ns5
import OpenElectrophy as OE
import numpy as np
import argparse
from os import path

# Constant of quantization
uV_QUANTUM = 4096. / 32768


def load_control_data():
    """Loads the control files from disk."""
    
    # How much of a slice should we take around each timestamp?
    parser = argparse.ArgumentParser(description='Load binary data into OE')
    #parser.add_argument('filename', metavar='FILENAME', type=str, nargs=1)    
    parser.add_argument('--pre', type=float, default=0.25)
    parser.add_argument('--post', type=float, default=0.75)    
    parser.add_argument('--data', metavar='Path to raw ns5 file', 
        type=str)
    parser.add_argument('--db', metavar='Desired database filename',
        type=str, default='test.db')
    parser.add_argument('--db_type', metavar='Desired database type',
        type=str, default='sqlite')        
    args = parser.parse_args()
    pre_slice_len = args.pre
    post_slice_len = args.post
    filename = args.data
    db_name = args.db
    db_type = args.db_type
    #filename = args.filename[0]
    
    data_dir = path.split(filename)[0]
    
    # Which channels should we load?
    try:
        SHOVE_CHANNELS = np.loadtxt(\
            path.join(data_dir, 'SHOVE_CHANNELS'), dtype=np.int)
    except IOError:
        # Should probably exit, rather than loading all channels
        print("CANNOT LOAD SHOVE_CHANNELS")
        SHOVE_CHANNELS = np.array([])
    
    # Which time slices should we load?
    try:
        TIMESTAMPS = np.loadtxt(\
            path.join(data_dir, 'TIMESTAMPS'), dtype=np.int)
    except IOError:
        # Should gracefully load all data
        raise IOError
        #TIMESTAMPS = np.array([])
    
    return (TIMESTAMPS, SHOVE_CHANNELS, pre_slice_len, post_slice_len,
        filename, db_name, db_type)

def stuff(filename, db_name, TIMESTAMPS, SHOVE_CHANNELS, 
    pre_slice_len, post_slice_len, db_type='sqlite'):
    
    # Load the file and file header
    l = ns5.Loader(filename=filename)    
    l.load_file()
    
    # Audio channel numbers
    AUDIO_CHANNELS = l.get_audio_channel_numbers()
    
    # Open connection to OE db and create a block
    if db_type is 'postgres':
        OE.open_db(url=('postgresql://postgres@192.168.0.171/test'))# %s' % db_name))
        print('post')
    else:
        OE.open_db(url=('sqlite:///%s' % db_name))
    #OE.open_db(url=('mysql://root:data@localhost/%s' % db_name))
    block = OE.Block(name='Raw Data', 
        info='Raw data sliced around trials',
        fileOrigin=filename)
    id_block = block.save() # need this later
    
    # Convert requested slice lengths to samples
    pre_slice_len_samples = int(pre_slice_len * l.header.f_samp)
    post_slice_len_samples = int(post_slice_len * l.header.f_samp)    
    
    # Add RecordingPoint
    ch2rpid = dict()
    for ch in SHOVE_CHANNELS:
        rp = OE.RecordingPoint(name=('RP%d' % ch), 
            id_block=id_block,
            channel=float(ch))
        rp_id = rp.save()
        ch2rpid[ch] = rp_id
    
    # Extract each trial as a segment
    for tn, trial_start in enumerate(TIMESTAMPS):
        # Create segment for this trial
        segment = OE.Segment(id_block=id_block, name=('trial%d' % tn),
            info='raw data loaded from good channels')
        id_segment = segment.save()        

        # Create AnalogSignal for each channel
        for chn, ch in enumerate(SHOVE_CHANNELS):
            # Load
            x = np.array(l._get_channel(ch)[trial_start-pre_slice_len_samples:\
                trial_start+post_slice_len_samples])
            
            # Convert to uV
            x = x * uV_QUANTUM
            
            # Put in AnalogSignal and save to db
            sig = OE.AnalogSignal(signal=x,
                channel=float(ch),
                sampling_rate=l.header.f_samp,
                t_start=(trial_start-pre_slice_len_samples)/l.header.f_samp,
                id_segment=id_segment,
                id_recordingpoint=ch2rpid[ch],
                name=('Channel %d Trial %d' % (ch, tn)))   
            
            # Special processing for audio channels
            if ch == AUDIO_CHANNELS[0]:
                sig.name = ('L Speaker Trial %d' % tn)
            elif ch == AUDIO_CHANNELS[1]:
                sig.name = ('R Speaker Trial %d' % tn)
            
            # Save signal to database
            sig.save()
        
        
        # Handle AUDIO CHANNELS only slightly differently
        for ch in AUDIO_CHANNELS:
            # Load
            x = np.array(l._get_channel(ch)[trial_start-pre_slice_len_samples:\
                trial_start+post_slice_len_samples])

            # Special processing for audio channels
            if ch == AUDIO_CHANNELS[0]:
                sname = ('L Speaker Trial %d' % tn)
            elif ch == AUDIO_CHANNELS[1]:
                sname = ('R Speaker Trial %d' % tn)

            # Put in AnalogSignal and save to db
            sig = OE.AnalogSignal(signal=x,
                channel=float(ch),
                sampling_rate=l.header.f_samp,
                t_start=(trial_start-pre_slice_len_samples)/l.header.f_samp,
                id_segment=id_segment,
                name=sname)
            
            # Save signal to database
            sig.save()
        


        # Save segment (with all analogsignals) to db
        # Actually this may be unnecessary
        # Does saving the signals link to the segment automatically?
        segment.save()
        
    return (id_segment, id_block)



(TIMESTAMPS, SHOVE_CHANNELS, pre_slice_len, 
    post_slice_len, filename, db_name, db_type) = load_control_data()

id = stuff(filename, db_name, TIMESTAMPS, SHOVE_CHANNELS, 
    pre_slice_len, post_slice_len, db_type)


