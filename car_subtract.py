# The purpose of this script is to re-reference each neural channel
# in the database. We use a common-average reference (CAR).
# Potentially I might want to go ahead and high pass filter at this
# point too.
#
# We'll insert the re-referenced signals back into a different block
# of the same database.
#
# A required input is "TETRODE_CHANNELS". This plaintext datafile contains
# the channels of each tetrode, and excludes all bad/broken channels.
# Only the channel numbers in this file will be used to compute the CAR!
#
# We'll also add the tetrode-specific information (trodness, group)
# at this point.

import OpenElectrophy as OE
import numpy as np
from os import path
import scipy.signal

#~ # For the moment, hard-code these filenames. How do we want to handle
#~ # this? It should probably except either a command line parameter, or
#~ # be callable from another function.
#~ #DB_NAME = 'test.db'


def get_tetrode_channels(filename):
    """ Load TETRODE_CHANNELS control file
    TODO: make sure this works for case of 1-trodes
    This is funny-looking because not all tetrodes have same number
    of channels!
    """
    f = file(filename)
    x = f.readlines()
    f.close()
    return [[int(c) for c in r] for r in [str.split(rr) for rr in x]]


def define_spike_filter(fs):
    filter_ord = 3
    low_cut = 300. / (fs / 2.)
    
    return scipy.signal.butter(filter_ord, low_cut, btype='high')

def define_spike_filter_2(fs):
    filter_ord = 3
    high_cut = 3000. / (fs / 2.)
    
    return scipy.signal.butter(filter_ord, high_cut, btype='low')


def run(db_name, CAR=True, smooth_spikes=True):
    """Filters the data for spike extraction.
    
    db_name: Name of the OpenElectrophy db file
    CAR: If True, subtract the common-average of every channel.
    smooth_spikes: If True, add an additional low-pass filtering step to
        the spike filter.
    """
    # Open connection to the database
    OE.open_db(url=('sqlite:///%s' % db_name))    
    
    # Check that I haven't already run
    id_blocks, = OE.sql("SELECT block.id FROM block WHERE block.name='CAR Tetrode Data'")
    if len(id_blocks) > 0:
        print "CAR Tetrode Data already exists, no need to recompute"
        return
    
    # Find the block
    id_blocks, = OE.sql("SELECT block.id FROM block WHERE block.name='Raw Data'")
    assert(len(id_blocks) == 1)
    id_block = id_blocks[0]
    raw_block = OE.Block().load(id_block)
    
    # Define spike filter
    # TODO: fix so that doesn't assume all sampling rates the same!
    fixed_sampling_rate = OE.AnalogSignal().load(1).sampling_rate
    FILTER_B, FILTER_A = define_spike_filter(fixed_sampling_rate)
    
    # If requested, define second spike filter
    if smooth_spikes is True:
        FILTER_B2, FILTER_A2 = define_spike_filter_2(fixed_sampling_rate)
    
    # Find TETRODE_CHANNELS file in data directory of db
    data_dir = path.split(raw_block.fileOrigin)[0]
    TETRODE_CHANNELS = get_tetrode_channels(path.join(data_dir,
        'TETRODE_CHANNELS'))
    N_TET = len(TETRODE_CHANNELS)

    # For convenience, flatten TETRODE_CHANNELS to just get worthwhile channels
    GOOD_CHANNELS = [item for sublist in TETRODE_CHANNELS for item in sublist]
    
    # Create a new block for referenced data, and save to db.
    car_block = OE.Block(\
        name='CAR Tetrode Data',
        info='Raw neural data, now referenced and ordered by tetrode',
        fileOrigin=db_name)
    id_car_block = car_block.save()

    # Make RecordingPoint for each channel, linked to tetrode number with `group`
    # Also keep track of link between channel and RP with ch2rpid dict
    ch2rpid = dict()
    for tn, ch_list in enumerate(TETRODE_CHANNELS):
        for ch in ch_list:
            rp = OE.RecordingPoint(name=('RP%d' % ch), 
                id_block=id_car_block,
                trodness=len(ch_list),
                channel=float(ch),
                group=tn)
            rp_id = rp.save()
            ch2rpid[ch] = rp_id

    # Find all segments in the block of raw data
    id_segments, = OE.sql('SELECT segment.id FROM segment ' + \
        'WHERE segment.id_block = :id_block', id_block=id_block)

    # For each segment in this block, load each AnalogSignal listed in
    # TETRODE channels and average
    # to compute CAR. Then subtract from each AnalogSignal.
    for id_segment in id_segments:   
        # Create a new segment in the new block with the same name
        old_seg = OE.Segment().load(id_segment)    
        car_seg = OE.Segment(name=old_seg.name,
            id_block=id_car_block,)
        id_car_seg = car_seg.save()
        
        # Find all AnalogSignals in this segment
        id_sigs, = OE.sql('SELECT analogsignal.id FROM analogsignal ' + \
            'WHERE analogsignal.id_segment = :id_segment', id_segment=id_segment)
        
        # Compute average of each
        running_car = 0
        n_summed = 0
        for id_sig in id_sigs: 
            sig = OE.AnalogSignal().load(id_sig)
            if sig.channel not in GOOD_CHANNELS:
                continue
            running_car = running_car + sig.signal
            n_summed = n_summed + 1
        
        # Zero out CAR if CAR is not wanted
        # TODO: eliminate the actual calculation of CAR above in this case
        # For now, just want to avoid weird bugs
        if CAR is False:
            running_car = np.zeros(running_car.shape)
        
        # Put the CAR into the new block
        # not assigning channel, t_start, sample_rate, maybe more?
        car_sig = OE.AnalogSignal(name='CAR',
            signal=running_car/n_summed,
            info='CAR calculated from good channels for this segment',
            id_segment=id_segment)    
        car_sig.save()

        # Put all the substractions in id_car_seg
        for id_sig in id_sigs:
            # Load the raw signal (skip bad channels)
            sig = OE.AnalogSignal().load(id_sig)
            if sig.channel not in GOOD_CHANNELS:
                continue
            
            # Subtract the CAR
            referenced_signal = sig.signal - car_sig.signal
            
            # Filter!
            filtered_signal = scipy.signal.filtfilt(FILTER_B, FILTER_A, 
                referenced_signal)
            if smooth_spikes is True:
                filtered_signal = scipy.signal.filtfilt(FILTER_B2, FILTER_A2,
                    filtered_signal)
            
            # Check for infs or nans
            if np.isnan(filtered_signal).any():
                print "ERROR: Filtered signal contains NaN!"
            if np.isinf(filtered_signal).any():
                print "ERROR: Filtered signal contains Inf!"
            
            # Store in db
            new_sig = OE.AnalogSignal(\
                name=sig.name,
                signal=filtered_signal,
                info='CAR has been subtracted',
                id_segment=id_car_seg,
                id_recordingpoint=ch2rpid[sig.channel],
                channel=sig.channel,
                t_start=sig.t_start,
                sampling_rate=sig.sampling_rate)
            new_sig.save()
        
        # Finally, copy the audio channel over from the old block
        id_audio_sigs, = OE.sql('SELECT analogsignal.id FROM analogsignal ' + \
            'WHERE analogsignal.id_segment = :id_segment AND ' + \
            "analogsignal.name LIKE '% Speaker %'", id_segment=id_segment)
        for id_audio_sig in id_audio_sigs:
            old_sig = OE.AnalogSignal().load(id_audio_sig)
            OE.AnalogSignal(\
                name=old_sig.name,
                signal=old_sig.signal,
                id_segment=id_car_seg,
                channel=old_sig.channel,
                t_start=old_sig.t_start,
                sampling_rate=old_sig.sampling_rate).save()


if __name__ is '__main__':
    # run argparse
    # call run()
    print "failed"