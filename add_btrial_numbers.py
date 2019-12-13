from __future__ import print_function
from __future__ import absolute_import
# Use the foil objects to correlate TIMESTAMPS (detected with cal code
# on 1-27-11) with the original ns5 file. Put the stim numbers into
# the database.

from builtins import zip
from builtins import object
from . import DataSession
import numpy as np
import os.path
from . import bcontrol
import OpenElectrophy as OE
import re

def run(control_params, auto_validate=True, v2_behavior=False):
    # Location of data
    data_dir = control_params['data_dir']

    # Location of the Bcontrol file
    bdata_filename = control_params['behavior_filename']

    # Location of TIMESTAMPS
    timestamps_filename = os.path.join(data_dir, 'TIMESTAMPS')

    # Location of OE db
    db_filename = control_params['db_name']

    # Load timestamps calculated from audio onsets in ns5 file
    ns5_times = np.loadtxt(timestamps_filename, dtype=np.int)

    # Load bcontrol data (will also validate)
    bcl = bcontrol.Bcontrol_Loader(filename=bdata_filename,
        auto_validate=auto_validate, v2_behavior=v2_behavior)
    bcl.load()

    # Grab timestamps from behavior file
    b_onsets = bcl.data['onsets']

    # Try to convert this stuff into the format expected by the syncer
    class fake_bcl(object):
        def __init__(self, onsets):
            self.audio_onsets = onsets
    class fake_rdl(object):
        def __init__(self, onsets):
            self.audio_onsets = onsets

    # Convert into desired format, also throwing away first behavior onset
    # We need to correct for this later.
    fb = fake_bcl(b_onsets[1:])
    fr = fake_rdl(ns5_times)

    # Sync. Will write CORR files to disk.
    # Also produces bs.map_n_to_b_masked and vice versa for trial mapping
    bs = DataSession.BehavingSyncer()
    bs.sync(fb, fr, force_run=True)

    # Put trial numbers into OE db
    db = OE.open_db('sqlite:///%s' % db_filename)

    # Each segment in the db is named trial%d, corresponding to the
    # ordinal TIMESTAMP, which means neural trial time.
    # We want to mark it with the behavioral trial number.
    # For now, put the behavioral trial number into Segment.info
    # TODO: Put the skip-1 behavior into the syncer so we don't have to
    # use the trick. Then we can use map_n_to_b_masked without fear.
    # Note that the 1010 data is NOT missing the first trial.
    # Double check that the neural TIMESTAMP matches the value in peh.
    # Also, add the check_audio_waveforms functionality here so that it's
    # all done at once.
    id_segs, name_segs = OE.sql('select segment.id, segment.name from segment')
    for id_seg, name_seg in zip(id_segs, name_segs):
        # Extract neural trial number from name_seg
        n_trial = int(re.search('trial(\d+)', name_seg).group(1))
        
        # Convert to behavioral trial number
        # We use the 'trial_number' field of TRIALS_INFO
        # IE the original Matlab numbering of the trial
        # Here we correct for the dropped first trial.
        try:
            b_trial = bcl.data['TRIALS_INFO']['TRIAL_NUMBER'][\
                bs.map_n_to_b_masked[n_trial] + 1]
        except IndexError:
            # masked trial
            if n_trial == 0:
                print("WARNING: Assuming this is the dropped first trial")
                b_trial = bcl.data['TRIALS_INFO']['TRIAL_NUMBER'][0]
            else:
                print("WARNING: can't find trial")
                b_trial = -99
        
        # Store behavioral trial number in the info field
        seg = OE.Segment().load(id_seg)
        seg.info = '%d' % b_trial
        seg.save()
