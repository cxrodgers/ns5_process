# Iterates through each trial, grabs audio waveform, checks against
# other waveforms of same type

import OpenElectrophy as OE
import numpy as np
import bcontrol
import matplotlib.pyplot as plt

def safe_index(l, val):
    try:
        return l.index(val)
    except ValueError:
        return np.nan


def select_segments_by_trial_number(id_block, trial_numbers):    
    # Load all segments from block
    id_segs, info_segs = OE.sql('select segment.id, segment.info from segment \
        where segment.id_block = %d' % id_block)
    
    # Extract b_trial_numbers in more useful format by strippin leading 'B'
    b_trial_numbers = info_segs.astype(int) #[int(info) for info in info_segs] 
    
    # Find id_segs that match these trials
    return id_segs[np.in1d(b_trial_numbers, trial_numbers)]


def select_audio_signals_by_stimulus_number(id_block, stim_num, 
    TRIALS_INFO, side='L'):
    # Find trials with stimulus number stim_num
    keep_id_segs = select_segments_by_trial_number(id_block, 
        trial_numbers=TRIALS_INFO['TRIAL_NUMBER'][\
        TRIALS_INFO['STIM_NUMBER'] == stim_num])

    # Load all analogsignals of this channel
    id_sigs, id_segs = OE.sql('select analogsignal.id, analogsignal.id_segment \
        from analogsignal where analogsignal.name like "' \
        + side + ' Speaker %"')
    
    # Grab analog signals where id_segs matches keep_id_segs    
    keep_id_sigs = id_sigs[np.in1d(id_segs.astype(int), 
        keep_id_segs.astype(int))]
    speaker_traces = [OE.AnalogSignal().load(id_sig).signal \
        for id_sig in keep_id_sigs]
    
    return np.array(speaker_traces)


def execute(control_params):
    # Load TRIALS_INFO
    bcl = bcontrol.Bcontrol_Loader(filename=control_params['behavior_filename'],
        v2_behavior=True)
    bcl.load()
    TRIALS_INFO = bcl.data['TRIALS_INFO']
    
    # Open database
    OE.open_db('sqlite:///%s' % control_params['db_name'])    
    id_blocks, = OE.sql('select block.id from block where block.name = "Raw Data"')
    id_block = id_blocks[0]

    pre_stim_len = int(control_params['pre_slice'] * 30000.)
    stim_len = int(.250 * 30000.)
    f1 = plt.figure(); f2 = plt.figure();
    l_sums = dict(); r_sums = dict();
    for sn in np.unique(TRIALS_INFO['STIM_NUMBER']):
        # Get all signals with certain stim number
        l_speaker_traces = select_audio_signals_by_stimulus_number(id_block,
            sn, TRIALS_INFO, 'L')
        r_speaker_traces = select_audio_signals_by_stimulus_number(id_block,
            sn, TRIALS_INFO, 'R')
        
        if sn == 6:
            return l_speaker_traces, r_speaker_traces
        
        ax = f1.add_subplot(3, 4, sn)
        ax.plot(l_speaker_traces[:, pre_stim_len + np.arange(-30, 30)].transpose())
        ax.set_title('L %d' % sn)
        
        ax = f2.add_subplot(3, 4, sn)
        ax.plot(r_speaker_traces[:, pre_stim_len + np.arange(-30, 30)].transpose())
        ax.set_title('R %d' % sn)
        
        slices = l_speaker_traces[:, pre_stim_len:pre_stim_len+stim_len]
        l_sums[sn] = 10*np.log10((slices.astype(np.float) ** 2).sum(axis=1))
        
        slices = r_speaker_traces[:, pre_stim_len:pre_stim_len+stim_len]
        r_sums[sn] = 10*np.log10((slices.astype(np.float) ** 2).sum(axis=1))
    
    plt.show()
    
    # Now plot powers
    plt.figure()
    plt.subplot(131)
    for sn in [1,2,3,4]:
        plt.plot(l_sums[sn], r_sums[sn], '.')
    plt.xlabel('left'); plt.ylabel('right')
    plt.legend(['lo', 'hi', 'le', 'ri'], loc='best')
    plt.title('Pure')
    
    plt.subplot(132)
    for sn in [5,6,7,8]:
        plt.plot(l_sums[sn], r_sums[sn], '.')
    plt.xlabel('left'); plt.ylabel('right')
    plt.legend(['le-hi', 'ri-hi', 'le-lo', 'ri-lo'], loc='best')
    plt.title('PB')
    
    plt.subplot(133)
    for sn in [9,10,11,12]:
        plt.plot(l_sums[sn], r_sums[sn], '.')
    plt.xlabel('left'); plt.ylabel('right')
    plt.legend(['le-hi', 'ri-hi', 'le-lo', 'ri-lo'], loc='best')
    plt.title('LB')
    plt.show()
    
    
    
