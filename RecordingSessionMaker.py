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
import ns5
import AudioTools
try:
    import OpenElectrophy as OE
except:
    print "cannot import OE"
    pass
import matplotlib
matplotlib.rcParams['figure.subplot.hspace'] = .5
matplotlib.rcParams['figure.subplot.wspace'] = .5
matplotlib.rcParams['font.size'] = 8.0
matplotlib.rcParams['xtick.labelsize'] = 'small'
matplotlib.rcParams['ytick.labelsize'] = 'small'
import matplotlib.pyplot as plt
import DataSession
import collections
import re
import SpikeTrainContainers
import bcontrol
import os.path
import shutil
from myutils import printnow, parse_bitstream
import pandas


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
        channel_groups_file, analog_channels_file=None):
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
        analog_channels_file : filename containing analog channels, or None
        
        All of the channel files should follow the format that
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
        
        # Read analog channels from disk
        if analog_channels_file is not None:
            self.analog_channel_ids = \
                RecordingSession.read_channel_numbers(analog_channels_file)[0]
        else:
            # No analog channels specified, none will be added to session
            self.analog_channel_ids = None
    
    def make_session(self, ns5_filename, session_name=None, remove_from_SC=[], 
        remove_from_TC=[]):
        """Create a RecordingSession for some data.
        
        ns5_filename : full path to raw ns5 file
        session_name : name of subdirectory, eg '110413_LBPB'
        remove_from_SC : channels to remove from SHOVE_CHANNELS
        remove_from_TC : channels to remove from TETRODE_CHANNELS

        Returns the created RecordingSession.
        """
        # Make default session name
        if session_name is None:
            session_name = os.path.splitext(
                os.path.split(os.path.normpath(ns5_filename))[1])[0]

        # Create RecordingSession in path defined by parameters
        full_path = os.path.join(self.data_analysis_dir, session_name)
        rs = RecordingSession.RecordingSession(full_path)
        
        # Link neural data to session dir, overwriting if necessary
        if os.path.isfile(ns5_filename):
            link_file(ns5_filename, rs.full_path)
        else:
            raise(IOError("%s is not a file" % ns5_filename))
        
        # Create channel numbering meta file
        #session_SC_list = sorted(list(set(self.SC_list) - set(remove_from_SC)))
        session_SC_list = [val for val in self.SC_list 
            if val not in remove_from_SC]
        rs.write_neural_channel_ids(session_SC_list)
        
        # Analog channels
        if self.analog_channel_ids is not None:
            rs.write_analog_channel_ids(self.analog_channel_ids)
        
        # Create channel grouping meta file
        #session_TC_list = [sorted(list(set(this_ch) - set(remove_from_TC)))\
        #    for this_ch in self.TC_list]
        session_TC_list = [[val for val in this_ch if val not in remove_from_TC]
            for this_ch in self.TC_list]
        rs.write_channel_groups(session_TC_list)
    
        return rs

# Many of the following functions work like wrapper for RecordingSession.
# Plot average LFP over neural channels
def plot_avg_lfp(rs, event_name='Timestamp', meth='avg', savefig=None,
    t_start=None, t_stop=None, harmonize_axes=True):
    """Plot average of analogsignals triggered on some event.
    
    Each channel gets its own subplot.
    
    meth : if 'avg', plot avg trace, if 'all', plot all traces
    harmonize_axes : keep x and y axes same for all subplots. Minimal
        overlap is chosen for x. Maximal range is chosen for y.
    """
    event_list = query_events(rs, event_name)
    if len(event_list) == 0:
        return
    
    # test ordering error
    # event_list = list(np.take(event_list, np.arange(len(event_list))[::-1]))

    # Make big figure with enough subplots
    f = plt.figure(figsize=(12,12))
    ymin, ymax, tmin, tmax = 0., 0., -np.inf, np.inf
    n_subplots = float(len(rs.read_neural_channel_ids()))
    spx = int(np.ceil(np.sqrt(n_subplots)))
    spy = int(np.ceil(n_subplots / float(spx)))
    
    # Call RecordingSession.avg_over_list_of_events for each channel
    for n, chn in enumerate(rs.read_neural_channel_ids()):
        ax = f.add_subplot(spx, spy, n+1)
        t, sig = rs.avg_over_list_of_events(event_list, chn=chn, meth=meth,
            t_start=t_start, t_stop=t_stop)
        ax.plot(t*1000., sig.transpose())
        
        # Get nice tight harmonized boundaries across all subplots
        if sig.min() < ymin: ymin = sig.min()
        if sig.max() > ymax: ymax = sig.max()        
        if t.min() > tmin: tmin = t.min()
        if t.max() < tmax: tmax = t.max()
        
        plt.title('ch %d' % chn)
    
    # Harmonize x and y limits across subplots
    if harmonize_axes:
        for ax in f.get_axes():        
            ax.set_xlim((tmin*1000., tmax*1000.))
            ax.set_ylim((ymin, ymax))
    
    if savefig is None:
        plt.show()
    elif savefig is True:
        filename = os.path.join(rs.full_path, rs.session_name + '_avglfp.png')
        plt.savefig(filename)
        plt.close()
    else:
        plt.savefig(savefig)
        plt.close()

def plot_all_spike_psths(rs, savefig=None):
    """Dump PSTHs of all spikes from each tetrode to disk.
    
    Does not use clustering information even if it exists.
    
    Currently doesn't work for any use case other than hard limits around
    each time stamp.
    
    This function handles arrangement into figure, but actual plotting
    is a method of PSTH.
    """
    f = plt.figure()
    #ymin, ymax, tmin, tmax = 0., 0., -np.inf, np.inf
    
    # Grab psths combined
    psths = rs.get_psths(combine_units=True)
    
    # subplot info
    n_subplots = len(psths)
    spx = int(np.ceil(np.sqrt(n_subplots)))
    spy = int(np.ceil(n_subplots / float(spx)))
    
    # get tetrode numbers, which are keys to psths
    tetnums = sorted(psths.keys())
    
    # plot each
    for n, tetnum in enumerate(tetnums):
        ax = f.add_subplot(spx, spy, n+1)
        psths[tetnum].plot(ax=ax)
        plt.title('all spikes from tet %d' % tetnum)
    
    # save
    if savefig is None:
        plt.show()
    elif savefig is True:
        filename = os.path.join(rs.full_path, rs.session_name + 
            '_unsorted_psth.png')
        f.savefig(filename)
        plt.close()        
    else:
        f.savefig(savefig)
        plt.close()

def convert_trial_numbers_to_segments(rs, trial_numbers, block='spike'):
    """Returns a list of Segment with info field matching trial numbers"""
    l2 = [str(tn) for tn in trial_numbers]
    if block == 'spike':
        id_block = rs.get_spike_block().id
    elif block == 'raw':
        id_block = rs.get_raw_data_block().id
    else:
        raise(ValueError("block must be 'spike' or 'raw'"))
    
    
    session = self.get_OE_session()
    seglist = session.query(OE.Segment).filter(
        OE.Segment.id_block == id_block).filter(
        OE.Segment.info.in_(l2)).all()
    
    return seglist
    

def plot_spike_rate_over_session(rs, savefig=None, skipNoScore=True):
    """Plot spike rate over session of MUA and SUA in spike picker.
    
    Bug: since this plots multiple figures, you can't specify a fixed
    filename in savefig or it will be overwritten. Leave as None or True.
    """
    sp = rs.get_spike_picker()
    
    # subplot info
    tetnums = sorted(sp.tetrodes)
    n_subplots = len(tetnums)
    spx = int(np.ceil(np.sqrt(n_subplots)))
    spy = int(np.ceil(n_subplots / float(spx)))
    
    # get block boundaries if possible
    session = rs.get_OE_session()
    te_list = []
    for tn in np.arange(80, 1200, 80).astype(np.int):
        q = session.query(OE.Segment).filter(OE.Segment.info == str(tn))
        try:
            te = q.first()._events[0].time
        except:
            te = None
        if te is not None:
            te_list.append(te)
    
    # plot each MUA
    f = plt.figure(figsize=(12,12))
    for n, tetnum in enumerate(tetnums):
        ax = f.add_subplot(spx, spy, n+1)
        spike_times = sp.pick_spikes(tetrode=[tetnum], adjusted=False)
        psth = SpikeTrainContainers.PSTH(adjusted_spike_times=spike_times, 
            binwidth=2., n_trials=1, F_SAMP=30000., 
            range=[spike_times.min(), spike_times.max()])
        psth.plot(ax, units='Hz')
        plt.title('all spikes from tet %d' % tetnum)
        for te in te_list:
            plt.plot([te], [0], '*', ms=18)
    
    # save
    if savefig is None:
        plt.show()
    elif savefig is True:
        filename = os.path.join(rs.full_path, rs.session_name + 
            '_MUA_FR_over_session.png')
        f.savefig(filename)
        plt.close()        
    else:
        f.savefig(savefig)
        plt.close()    
    
    # get units to plot
    if skipNoScore:
        session = rs.get_OE_session()
        nl = session.query(OE.Neuron).filter(OE.Neuron.sortingScore != None).all()
        good_SU_list = [n.id for n in nl]
    else:
        # Plot all
        good_SU_list = sp.units    
    
    # subplot info
    n_subplots = len(good_SU_list)
    if n_subplots == 0:
        return
    spx = int(np.ceil(np.sqrt(n_subplots)))
    spy = int(np.ceil(n_subplots / float(spx)))
    
    # plot each SU
    f = plt.figure(figsize=(16, 12))
    for n, unit in enumerate(good_SU_list):
        ax = f.add_subplot(spx, spy, n+1)
        spike_times = sp.pick_spikes(unit=[unit], adjusted=False)
        psth = SpikeTrainContainers.PSTH(adjusted_spike_times=spike_times, 
            binwidth=2., n_trials=1, F_SAMP=30000., 
            range=[spike_times.min(), spike_times.max()])
        psth.plot(ax, units='Hz')
        plt.title('all spikes from unit %d' % unit)
        for te in te_list:
            plt.plot([te], [0], '*', ms=18)
    
    # save
    if savefig is None:
        plt.show()
    elif savefig is True:
        filename = os.path.join(rs.full_path, rs.session_name + 
            '_SUA_FR_over_session.png')
        f.savefig(filename)
        plt.close()        
    else:
        f.savefig(savefig)
        plt.close()    


def plot_all_spike_psths_by_stim(rs, savefig=None, t_start=None, t_stop=None,
    skipNoScore=True, binwidth=.010, specify_unit_ids=None,
    check_against_trial_slicer=True, override_path=None):
    """Dump PSTHs of all SUs to disk, arranged by stimulus.
    
    This function handles arrangement into figure, but actual plotting
    is a method of PSTH.
    
    skipNoScore : if True, queries OE db for each unit id and only handles
        ones that have been assigned a score. will error if not found in db.
    
    specify_units_ids : plot only the requested units, this is mutually
        exclusive with skipNoScore
    
    override_path, check_against_trial_slicer : see rs.get_spike_picker
        Also, will save figures in override_path. Specify full path!
    """
    # Load trial info
    bcld = bcontrol.Bcontrol_Loader_By_Dir(rs.full_path)
    bcld.load()
    sn2trials = bcld.get_sn2trials()
    sn2names = bcld.get_sn2names()

    # Load spike picker
    sp = rs.get_spike_picker(override_path=override_path,
        check_against_trial_slicer=check_against_trial_slicer)
    
    if override_path:
        figpath = override_path
    else:
        figpath = rs.full_path
    
    # Choose units to plot
    if skipNoScore:
        session = rs.get_OE_session()
        nl = session.query(OE.Neuron).filter(OE.Neuron.sortingScore != None).all()
        good_SU_list = [n.id for n in nl]
    elif specify_unit_ids:
        good_SU_list = np.asarray(specify_unit_ids)
    else:
        # Plot all
        good_SU_list = sp.units
    
    for unit in good_SU_list:
        if unit not in sp.units:
            raise ValueError(
                "cannot find unit %d in spike times, try re-dumping")
        f = plt.figure(figsize=(16,12))
        #ymin, ymax, tmin, tmax = 0., 0., -np.inf, np.inf
        for sn, name in sn2names.items(): 
            if sn not in sn2trials:
                continue
            ax = f.add_subplot(3, 4, sn)
            psth = sp.get_psth(unit=[unit], trial=sn2trials[sn], 
                binwidth=binwidth)
            psth.plot(ax, style='elastic')
            if t_start is not None and t_stop is not None:
                ax.set_xlim(t_start, t_stop)            
            plt.title(name)
            plt.suptitle('%s - unit %d' % (rs.session_name, unit))
    
        # save
        if savefig is None:
            plt.show()
        elif savefig is True:
            filename = os.path.join(figpath, rs.session_name + 
                '_psth_by_stim_unit_%d.png' % unit)
            f.savefig(filename)
            plt.close(f)        
        else:
            filename = os.path.join(figpath, rs.session_name + 
                ('_psth_by_stim_unit_%d_' % unit) + 
                savefig + '.png')       
            f.savefig(filename)
            plt.close(f)            

def plot_MUA_by_stim(rs, savefig=None, t_start=None, t_stop=None, 
    binwidth=.010, override_path=None):
    """Dump PSTHs of all spikes from each tetrode to disk, by stimulus.
    
    Does not use clustering information even if it exists.
    
    Currently doesn't work for any use case other than hard limits around
    each time stamp.
    
    This function handles arrangement into figure, but actual plotting
    is a method of PSTH.
    
    savefig : if True, then a filename is auto-generated and the figure
        is saved to the RecordingSession directory.
        if False, then the figure is displayed.
        otherwise, a filename is auto-generated and savefig is appended
        to this filename and then saved to the directory.
    t_start, t_stop : if both are not None, then the t limits of each
        axis will be changed to this.
    
    override_path : gets the spike picker from a different directory,
        for instance if you have multiple klusters
    """
    # Load trial info
    bcld = bcontrol.Bcontrol_Loader_By_Dir(rs.full_path)
    bcld.load()
    sn2trials = bcld.get_sn2trials()
    sn2names = bcld.get_sn2names()

    # Load spike picker
    #spt = rs.get_spike_picker()._p.dict_by('tetrode')
    sp = rs.get_spike_picker(override_path=override_path)
    
    
    for tetnum in sorted(sp.tetrodes):
        
        f = plt.figure(figsize=(16,12))
        #ymin, ymax, tmin, tmax = 0., 0., -np.inf, np.inf
        for sn, name in sn2names.items(): 
            if sn not in sn2trials or len(sn2trials[sn]) == 0:
                continue
            ax = f.add_subplot(3, 4, sn)
            psth = sp.get_psth(tetrode=[tetnum], trial=sn2trials[sn], 
                binwidth=binwidth)
            psth.plot(ax, style='elastic')
            if t_start is not None and t_stop is not None:
                ax.set_xlim(t_start, t_stop)
            plt.title(name)
            plt.suptitle('%s - tetrode %d' % (rs.session_name, tetnum))
    
        # save
        if savefig is None:
            plt.show()
        elif savefig is True:
            filename = os.path.join(rs.full_path, rs.session_name + 
                '_psth_by_stim_tet_%d.png' % tetnum)
            f.savefig(filename)
            plt.close()        
        else:
            filename = os.path.join(rs.full_path, rs.session_name + 
                ('_psth_by_stim_tet_%d_' % tetnum) + 
                savefig + '.png')       
            f.savefig(filename)
            plt.close(f)

# Plot average audio signal
def plot_avg_audio(rs, event_name='Timestamp', meth='all', savefig=None,
    t_start=None, t_stop=None):
    """Plot average of analogsignals triggered on some event.
    
    Each channel gets its own subplot.
    """
    event_list = query_events(rs, event_name)
    if len(event_list) == 0:
        return
    
    # test ordering error
    # event_list = list(np.take(event_list, np.arange(len(event_list))[::-1]))

    # Call RecordingSession.avg_over_list_of_events for each channel
    for n, chn in enumerate([chn + 128 for chn in rs.read_analog_channel_ids()]):
        plt.figure()
        t, sig = rs.avg_over_list_of_events(event_list, chn=chn, meth=meth,
            t_start=t_start, t_stop=t_stop)
        plt.plot(t*1000, sig.transpose())
        plt.title('ch %d' % chn)
    
        if savefig is None:
            plt.show()
        elif savefig is True:
            filename = os.path.join(rs.full_path, rs.session_name + ('_audio%d.png' % chn))
            plt.savefig(filename)
            plt.close()
        else:
            plt.savefig(savefig)
            plt.close()

# Plot average PSD
def plot_psd_by_channel(rs, event_name='Timestamp', meth='avg_db',
    harmonize_axes=True, savefig=None, f_range=(None, None), NFFT=2**12,
    normalization=0.0):
    """Plot PSD of each channel"""
    # get events
    event_list = query_events(rs, event_name)
    if len(event_list) == 0:
        return
    
    # Make big figure with enough subplots
    f = plt.figure(figsize=(12,12))
    ymin, ymax, tmin, tmax = np.inf, -np.inf, np.inf, -np.inf
    n_subplots = float(len(rs.read_neural_channel_ids()))
    spx = int(np.ceil(np.sqrt(n_subplots)))
    spy = int(np.ceil(n_subplots / float(spx)))

    # plot each channel
    for n, chn in enumerate(rs.read_neural_channel_ids()):
        ax = f.add_subplot(spx, spy, n+1)
        
        # calculate and plot PSD for this channel
        siglist = rs.get_signal_list_from_event_list(event_list, chn)
        if len(siglist) == 0:
            continue
        spectra, freqs = rs.spectrum(signal_list=siglist, meth='avg_db', 
            NFFT=NFFT, normalization=normalization)
        ax.semilogx(freqs, spectra, '.-')
        ax.grid()
        
        # Get nice tight harmonized boundaries across all subplots
        if spectra.min() < ymin: ymin = spectra.min()
        if spectra.max() > ymax: ymax = spectra.max()        
        if freqs.min() < tmin: tmin = freqs.min()
        if freqs.max() > tmax: tmax = freqs.max()
        
        plt.title('ch %d' % chn)
    plt.suptitle(rs.session_name + ' each psd')

    # Harmonize x and y limits across subplots
    if harmonize_axes:
        for ax in f.get_axes():        
            ax.set_xlim((tmin, tmax))
            ax.set_ylim((ymin, ymax))
    
    if f_range != (None, None):
        for ax in f.get_axes():
            ax.set_xlim(f_range)
            # ylim will be too broad
    
    # save figure, or display
    if savefig is None:
        plt.show()
    elif savefig is True:
        filename = os.path.join(rs.full_path, rs.session_name + '_each_psd.png')
        plt.savefig(filename)
        plt.close()
    else:
        plt.savefig(savefig)
        plt.close()
    
    

def query_events(rs, event_name='Timestamp'):
    # Open db
    #rs.open_db()
    session = rs.get_OE_session()
    block = rs.get_raw_data_block()
    
    # Get list of events in this block and with this name
    # Order by id_segment
    event_list = session.query(OE.Event).\
        join((OE.Segment, OE.Event.segment)).\
        filter(OE.Segment.id_block == block.id).\
        filter(OE.Event.label == event_name).\
        order_by(OE.Segment.id).all()
    
    return event_list

# Function to load audio data from raw file and detect onsets
def add_timestamps_to_session(rs, force=False, drop_first_N_timestamps=0, 
    meth='audio_onset', verbose=False, save_trial_numbers=True, **kwargs):
    """Given a RecordingSession, makes TIMESTAMPS file.
    
    I'm a thin wrapper over `calculate_timestamps` that knows how to get
    and put the data from RecordingSession objects.
    
    If the timestamps file already exists, then I return them but don't
    re-calculate. (Unless force=True, in which case I always run.)
    
    I get the right channels and filenames from RecordingSession, then call
    `calculate_timestamps`, then tell the session what I've calculated.
   
    Return values depend on `meth`, but the first is always the onset times
    in samples.
   
    Acceptable values for `meth`:
        'audio_onset' : calls calculate_timestamps and either operates on
            audio directly, or trigger channel if specified (see below).
            This method reads audio channels from RS and passes them on.
            In this case the returned values are:
                onset_times, offset_times
        'digital_trial_number' : calls calculate_timestamps_from_digital
            and reads the digital trial numbers.
            In this cas ethe returned values are:
                onset_times, trial_numbers
            
            If the bcontrol file exists, it will be loaded and the time
            difference between audio onset and trial start time will be 
            accounted for.
            
            If save_trial_numbers, will write them in a text file called
            'TRIAL_NUMBERS' in the RS directory.
            
        In either case, kwargs is passed to those underlying methods, so
        see those docstrings for details.
    
    If trigger channel
        * Should be integer, not list
        * manual threshhold should be specified, use something like 15e3,
          not dB like other case
    
    All keyword arguments (except drop_first_N_timestamps) are passed
    to calculate_timestamps, so see documentation there.
        manual_threshhold=None, audio_channels=None, 
        minimum_duration_ms=50, pre_first=0, post_last=0, debug_mode=False, 
        verbose=False, trigger_channel=None
    manual_threshhold should be in dB
    
    drop_first_N_timestamps : after calculation of timestamps, drop
    the first N of them before writing to disk. Another alternative is
    the pre_first and post_last kwargs which can be specified in samples.
    """
    # Check whether we need to run
    if not force and os.path.exists(os.path.join(rs.full_path, 
        RecordingSession.TIMESTAMPS_FILENAME)):
        
        try:
            known_trial_numbers = np.loadtxt(
                os.path.join(rs.full_path, 'TRIAL_NUMBERS'),
                dtype=np.int)
        except IOError:
            known_trial_numbers = []
        
        return (rs.read_timestamps(), known_trial_numbers)
    
    # Get data from recording session
    filename = rs.get_ns5_filename()
    
    if meth == 'audio_onset':
        # Detect audio onsets
        audio_channels = rs.read_analog_channel_ids()
        t_on, t_off = calculate_timestamps(filename, verbose=verbose,
            audio_channels=audio_channels, **kwargs)
        
        if drop_first_N_timestamps > 0:
            t_on = t_on[drop_first_N_timestamps:]
            t_off = t_off[drop_first_N_timestamps:]
        
        # Tell RecordingSession about them
        rs.add_timestamps(t_on)
        return (t_on, t_off)
        
    elif meth == 'digital_trial_number':
        trial_start_times, trial_numbers = \
            calculate_timestamps_from_digital_and_sync(rs, verbose=verbose, 
                drop_first_N_timestamps=drop_first_N_timestamps, **kwargs)

        rs.add_timestamps(trial_start_times)
        
        if save_trial_numbers:
            np.savetxt(os.path.join(rs.full_path, 'TRIAL_NUMBERS'),
                trial_numbers, fmt='%d')
        
        return trial_start_times, trial_numbers
        
    else:
        raise "unsupported method %s" % meth

def calculate_timestamps_from_digital_and_sync(rs, verbose=False, 
    skip_verification=False, assumed_dilation=.99663, wordlen=160, 
    drop_first_N_timestamps=0, **kwargs):
    """Calculates timestamps from digital signal and syncs with behavior file.
    
    Wrapper around `calculate_timestamps_from_digital` which does basic
    error checking on the parsed bitstream.
    
    This method additionally loads the bcontrol data and:
    * accounts for time between trial start (digital signal) and stimulus onset
    * drops timestamps that are not present in one or the other data source
    
    It also accounts for temporal dilation between these data sources.
    
    skip_verification : if True, doesn't attempt to load bcontrol data, just
        returns the raw result of the parse. In this case you might as well
        use `calculate_timestamps_from_digital`
    assumed_dilation : this is the value you get when you divide times from
        the neural recording computer, by times from the behavior recording
        computer. This is verified to be true for the loaded data, and
        then used to account for the time delay to onset.
    wordlen : length of the digital word in samples, which is accounted for
        in calculating stimulus onset
    drop_first_N_timestamps : drops this many timestamps from the beginning
    kwargs : passed to calculate_timestamps_from_digital (pre_first, post_last,
        manual_threshold, etc)
    
    Returns trial_start_times, trial_numbers
        trial_start_times is in samples and is the time of stimulus onset,
        rather than the time of the digital pulse.
    """
    # Parse the digital words
    filename = rs.get_ns5_filename()
    trial_start_times, trial_numbers = calculate_timestamps_from_digital(
        filename, verbose=verbose, **kwargs)
    if len(trial_start_times) == 0:
        if verbose:
            print "no times found!"
        return np.array([]), np.array([])
    
    # if possible, load behavioral data
    bcld = bcontrol.Bcontrol_Loader_By_Dir(rs.full_path)
    try:
        bcld._find_bcontrol_matfile()
    except:
        skip_verification = True
    
    # now use that behavioral data to account for difference between
    # trial start and onset
    if not skip_verification:
        if verbose:
            printnow("Checking for missing trials and verifying dilation")
        bcld.load()
        peh = bcld.data['peh']
        btrial_starts = np.array([trial['states']['state_0'][0,1] 
            for trial in bcld.data['peh']])
        bstim_onsets = np.array([trial['states']['play_stimulus'][0]
            for trial in bcld.data['peh']])

        # account for any missing trials
        if trial_numbers[0] != 1:
            vmsg = "%d trials occurred before recording started" % (
                trial_numbers[0] - 1)
            btrial_starts = btrial_starts[(trial_numbers[0] - 1):]
            bstim_onsets = bstim_onsets[(trial_numbers[0] - 1):]
        if len(trial_numbers) > len(btrial_starts):
            ndrop = len(trial_numbers) - len(btrial_starts)
            vmsg = "%d trials occurred after last " % ndrop + \
                "saved behavioral trial, dropping"
            trial_numbers = trial_numbers[:len(btrial_starts)]
            trial_start_times = trial_start_times[:len(btrial_starts)]
        elif len(btrial_starts) > len(trial_numbers):
            ndrop = len(btrial_starts) - len(trial_numbers)
            vmsg = "%d trials occurred after last trial in recording" % ndrop
            btrial_starts = btrial_starts[:len(trial_numbers)]
            bstim_onsets = bstim_onsets[:len(trial_numbers)]            
        if verbose:
            print vmsg

        # check that time syncs up    
        stretch_factors = np.diff(
            trial_start_times / rs.get_sampling_rate()) \
            / np.diff(btrial_starts)
        if np.max(np.abs(stretch_factors - assumed_dilation)) > .001:
            print "WARNING: dilation is off, suspect sync error"
        
        # now account for time to stimulus onset (including temporal dilation)
        if verbose:
            printnow("Converting digital timestamps into audio onsets")          
        stimulus_latencies = bstim_onsets - btrial_starts
        stimulus_latencies = stimulus_latencies * assumed_dilation
        
        # Convert to samples and add in the length of the digital word itself
        stimulus_latencies = np.rint(
            stimulus_latencies * rs.get_sampling_rate()).astype(np.int)
        trial_start_times = trial_start_times + stimulus_latencies
        trial_start_times = trial_start_times + wordlen
        
    elif verbose:
        print "cannot load bcontrol data, skipping verification"

    # write to directory
    if drop_first_N_timestamps > 0:
        trial_start_times = trial_start_times[drop_first_N_timestamps:]
        trial_numbers = trial_numbers[drop_first_N_timestamps:]
    
    return trial_start_times, trial_numbers


def calculate_timestamps(filename, manual_threshhold=None, audio_channels=None, 
    minimum_duration_ms=50, pre_first=0, post_last=0, debug_mode=False, 
    verbose=False, trigger_channel=None):
    """Given ns5 file and channel numbers, returns audio onsets.
    
    I uses `ns5.Loader` to open ns5 file and `AudioTools.OnsetDetector`
    to identify onsets. I discard onsets too close to the ends.
    
    Inputs
    ------
    filename : path to *.ns5 binary file
    manual_threshhold, minimum_duration_ms : Passed to OnsetDetector   
        manual_threshhold should be in dB 
    audio_channels : the channel numbers to analyze for onsets. If None,
        I'll use the first channel, or the first two channels if available.    
    pre_first : audio data before this sample will be ignored; therefore,
        the first onset will be at least this.
    post_last : amount of data to ignore from the end. Uses normal indexing
        rules: if positive, then this is the last sample to analyze;
        if negative, then that many samples from the end will be ignored;
        if zero, then all data will be included (default).
    debug_mode : Will plot the audio data before analyzing it.
    
    Returns
    ------
    Tuple of detected onsets and detected offsets in samples
    """
    # Load ns5 file
    l = ns5.Loader(filename=filename)
    l.load_file()
    
    # Which audio channels exist in the database?
    existing_audio_channels = l.get_analog_channel_ids()
    if len(existing_audio_channels) == 0:
        raise(IOError("No audio data exists in file %s" % filename))
    
    # If no audio channels were specified, use the first one or two
    if audio_channels is None:
        audio_channels = existing_audio_channels[0:2]
    
    if len(audio_channels) > 2: print "warning: >2 channel data is not tested"

    # Now create the numpy array containing mono or stereo data
    if trigger_channel is None:
        audio_data = np.array(\
            [l.get_analog_channel_as_array(ch) for ch in audio_channels])
    else:
        audio_data = np.array(\
            [l.get_analog_channel_as_array(trigger_channel)])
    
    # Slice out beginning and end of data
    if post_last == 0: 
        # Do not truncate end
        post_last = audio_data.shape[1]
    audio_data = audio_data[:, pre_first:post_last]

    # Plot the sliced data
    if debug_mode:
        import matplotlib.pyplot as plt
        plt.plot(audio_data.transpose())
        plt.show()

    # OnsetDetector expect mono to be 1d, not Nx1
    if len(audio_channels) == 1:
        audio_data = audio_data.flatten()

    # Instantiate an onset detector
    od = AudioTools.OnsetDetector(input_data=audio_data,
        F_SAMP=l.header.f_samp,
        manual_threshhold=manual_threshhold,
        minimum_duration_ms=minimum_duration_ms,
        verbose=verbose)

    if trigger_channel is None:
        # Run it and get the answer. Account for pre_first offset. Return.
        od.execute()
    else:
        # Override
        sound_bool = audio_data > manual_threshhold
        od._error_check_onsets(sound_bool)
    
    return (od.detected_onsets + pre_first, od.detected_offsets + pre_first)
    

def calculate_timestamps_from_digital(filename, trial_number_channel=16, 
    pre_first=0, post_last=0, debug_mode=False, verbose=True):
    """Calculates timestamps from a digital trial number signal
    
    The underlying work is done by parse_bitstream.
    This function loads the data from the specified channel and
    does some simple error checking and verbosity, as well as dropping
    trials from the start or end.
    
    filename : ns5 filename
    trial_number_channel : analog input containing the digital signal
    pre_first : data before this sample will be ignored; therefore,
        the first onset will be at least this.
    post_last : amount of data to ignore from the end. Uses normal indexing
        rules: if positive, then this is the last sample to analyze;
        if negative, then that many samples from the end will be ignored;
        if zero, then all data will be included (default).
    verbose : print detected trial numbers
    debug_mode : parse_bitstream will plot words
    
    Returns: trial_start_times, trial_numbers
    The first is the detected word onsets in samples from the beginning of
    the file. The second is the integer value of each word.
    """  
    # Load ns5 file
    l = ns5.Loader(filename=filename)
    l.load_file()
    
    # Get signal and truncate as requested
    trialnum_sig = l.get_analog_channel_as_array(trial_number_channel)
    if post_last == 0:
        # Do not truncate end
        post_last = len(trialnum_sig)
    trialnum_sig = trialnum_sig[pre_first:post_last]

    # Parse
    trial_start_times, trial_numbers = parse_bitstream(trialnum_sig,
        debug_mode=debug_mode)
    
    # Return empty if none found
    if len(trial_start_times) == 0:
        print "warning: no trial numbers found in %s" % filename
    else:
        # Some debugging stuff
        if verbose:
            print "found trials from %d to %d" % (
                trial_numbers[0], trial_numbers[-1])
        if np.any((trial_numbers - trial_numbers[0]) != 
            range(len(trial_numbers))):
            print "warning: trial numbers not ordered correctly"
        
        # Account for truncation
        trial_start_times = trial_start_times + pre_first
    
    return trial_start_times, trial_numbers


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

def add_behavioral_trial_numbers2(rs, known_trial_numbers=None,
    trial_number_channel=16, verbose=False):
    """Adds trial numbers to the Segment info field
    
    This is supposed to be a complete replacement for
        add_behavioral_trial_numbers
    because it includes new functionality for syncing with digital
    trial number signal. There is a wrapper which will fall back to that
    old method if necessary (ie, no digital signal available). In that case
    it will also write out TRIAL_NUMBERS, which is the new standard for
    converting from neural to behavioral trial number (replaces reading the
    OE info field).
    
    If you already know the behavioral numbers you can specify them as
    an attribute. In this case the length of this attribute should match
    the number of Segment in each Block. (That is, you must account for
    missing trials yourself.)
    
    If known_trial_numbers is None, will first look for a file called
    TRIAL_NUMBERS in the rs direcotry containing the numbers. Then it will
    attempt to load the trial numbers from the digital trial number signal 
    on channel `trial_number_channel`.
    
    If all else fails, will correlate the timestamps with the audio onsets
    in the bcontrol file. 
    """    
    session = rs.get_OE_session()
    
    # First try to load from plaintext
    if known_trial_numbers is None:
        try:
            known_trial_numbers = np.loadtxt(
                os.path.join(rs.full_path, 'TRIAL_NUMBERS'),
                dtype=np.int)
        except IOError:
            known_trial_numbers = None

    # Alternate methods
    if known_trial_numbers is None:
        chlist = rs.get_ns5_loader().get_analog_channel_ids()
        run_classic = True
        
        if trial_number_channel in chlist:
            if verbose:
                printnow("trying to get trial numbers from digital signal")
            # Try to get them from the digital signal
            trial_times, known_trial_numbers = \
                calculate_timestamps_from_digital_and_sync(rs, verbose=verbose,
                    trial_number_channel=trial_number_channel)

            # Check that we found enough trial numbers
            n_segments = len(session.query(OE.Block).first()._segments)
            if len(known_trial_numbers) == n_segments:
                run_classic = False
            elif verbose:
                printnow("detected trial numbers did not match # segments")
        
        if run_classic:
            # Wrapper around old sync methods
            # Calls add_behavioral_trial_numbers to do the syncing and
            # storing in OE.
            # So we don't need to do the OE store as below
            # However, we still need to save TRIAL_NUMBERS for compatibility
            # with current code which depends on it
            if verbose:
                printnow("falling back to classic detection")
            
            # This does the complicated syncing and stores in OE db
            add_behavioral_trial_numbers(rs)
            
            # Extract info field and write to TRIAL_NUMBERS in rs
            trial_numbers_from_info_field_to_text(rs, write_to_rs=True)
            
            # Previous version of code to do the same thing
            #~ trialnumbers = []
            #~ block = rs.get_raw_data_block()
            #~ for n, seg in enumerate(block._segments):
                #~ assert seg.name == ('Segment %d' % n)
                #~ bnum = int(seg.info)
                #~ trialnumbers.append(bnum)
            #~ np.savetxt(os.path.join(rs.full_path, 'TRIAL_NUMBERS'),
                #~ trialnumbers, fmt='%d')
            
            return
    
    # Here we actually add the trial numbers to each segment
    # We index into `known_trial_numbers` with the neural trial number
    if verbose:
        printnow("storing trial numbers")
    block_list = session.query(OE.Block).all()
    for block in block_list:
        for seg in block._segments:
            # Neural number of the trial
            n_trial = int(re.search('Segment (\d+)', seg.name).group(1))            
            seg.info = str(known_trial_numbers[n_trial])
            #seg.save() # this resaves the whole deal!
    session.commit()

def trial_numbers_from_info_field_to_text(rs, write_to_rs=True):
    """Parses trial numbers from OE db, converts to text TRIAL_NUMBERS
    
    This is a conversion method from an old way of storing behavioral trial
    numbers (in the info field of each Segment) to a new way (a simple
    flat text file consisting of each trial number).
    
    Also does some simple error checking
    1.  The trial numbers should be monotonically increasing. If this is
        not the case, then probably there is an issue with SQL truncation.
    2.  The values derived from the raw data block and spike data blocks
        should be the same.
    3.  The segments themselves should be named with the correct neural
        numbering. This is simply range(len(n_segments))
    """
    # Load btrial numbers from raw data
    trialnumbers = []
    block = rs.get_raw_data_block()
    for n, seg in enumerate(block._segments):
        # Check that the name encodes the neural trial number
        assert seg.name == ('Segment %d' % n)
        bnum = int(seg.info)
        trialnumbers.append(bnum)
    
    # Load btrial numbers from spike data (should be same)
    trialnumbers2 = []
    block = rs.get_spike_block()
    for n, seg in enumerate(block._segments):
        # Check that the name encodes the neural trial number        
        assert seg.name == ('Segment %d' % n)
        bnum = int(seg.info)
        trialnumbers2.append(bnum)
    
    # Convert to array
    trialnumbers = np.array(trialnumbers)
    trialnumbers2 = np.array(trialnumbers2)
    
    # Check consistency between blocks
    if not np.all(trialnumbers == trialnumbers2):
        print ("WARNING: spike block and raw block trial numbers differ in " +
            rs.full_path)
    
    # Check that the trial numbers look plausible
    if not np.all(np.diff(trialnumbers) == 1):
        print ("WARNING: trial numbers are not contiguous in " + rs.full_path)
    
    # Write out to the directory
    if write_to_rs:
        np.savetxt(os.path.join(rs.full_path, 'TRIAL_NUMBERS'),
            trialnumbers, fmt='%d')    
    
    return trialnumbers

def write_b2n_sync_file(rs, btimes=None, ntimes=None, peh=None,
    known_trial_numbers=None, extra_peh_entries=0, force=False,
    assumed_dilation=.99663):
    """Write syncing file for converting from behavior time to neural time.
    
    All this does is sync the behavioral times to the neural times with a 
    polynomial. The tricky part is accounting for missing trials. The easiest
    way to do this is to provide known_trial_numbers (or write the file
    called TRIAL_NUMBERS in rs.full_path). These are used to index into
    `parsed_events_history` from the bcontrol file.
    
    You can provide `btimes` and/or `ntimes` as is. (Specify in seconds)
    
    If you don't provide ntimes, it's exctracted from rs.read_timestamps()
    If you don't provide btimes, then I need peh and known_trial_numbers, which
    I can try to load from bcontrol file and from TRIAL_NUMBERS. In this case,
    I account for Matlab numbering: an entry of 1 in TRIAL_NUMBERS corresponds
    to peh[0]

    extra_peh_entries : I guess there is a possibility that there are extra
    peh states at the beginning, from before the trial numbering starts. In
    this case, I will add extra_peh_entries to the index into peh ... but note
    that this is not tested.
    
    assumed_dilation : asserts dilation is within .00005 of this, or errors.
        If you don't want to check this, set to None.

    Finally I write SYNC_B2N to the RS.
    """
    if not force and os.path.exists(os.path.join(rs.full_path, 'SYNC_B2N')):
        return
    
    # Account for indexing TRIAL_NUMBERS into peh
    peh_offset = -1 + extra_peh_entries 
    
    if btimes is None:
        # Calculate btimes from peh and known_trial_numbers
        if peh is None:
            bcld = bcontrol.Bcontrol_Loader_By_Dir(rs.full_path)
            bcld.load()
            peh = bcld.data['peh']
        
        if known_trial_numbers is None:
            # If it fails here, might be able to auto-guess the trials are
            # lined up
            known_trial_numbers = np.loadtxt(
                os.path.join(rs.full_path, 'TRIAL_NUMBERS'), dtype=np.int)
        
        # Use peh and known_trial_numbers to extract btimes
        btimes = []
        for trial_number in known_trial_numbers:
            trial = peh[trial_number + peh_offset]
            btimes.append(trial['states']['play_stimulus'][0])
        btimes = np.asarray(btimes)
    
    if ntimes is None:
        ntimes = rs.read_timestamps() / rs.get_sampling_rate()
    
    # fit
    b2n = np.polyfit(btimes, ntimes, deg=1)
    
    # error check
    if assumed_dilation is not None:
        if np.abs(b2n[0] - assumed_dilation) > .00005:
            raise ValueError("Sync error: got dilation of %f" % b2n[0])
        
    np.savetxt(os.path.join(rs.full_path, 'SYNC_B2N'), b2n)

def write_events_file(rs, known_trial_numbers=None, convert_to_ntime=True,
    force=False):
    """Write bcontrol events and trials_info as flat text file
    
    The bcontrol data is loaded from the bcontrol data file using
    Bcontrol_Loader_By_Dir on the recording session, so it must exist there.
    
    Set convert_to_ntime to True to convert event times to neural timebase.
    For this to work, the RS needs to have a SYNC_B2N file. 
    
    If SYNC_B2N does not exist:
    I'll call write_b2n_sync_file to generate one. That method needs to know
    when the neural onsets occurred and which trial each one was. I can
    pass `known_trial_numbers` to it, or it can read TRIAL_NUMBERS from the RS.
    
    There is space in this method to deal with the possibility of peh
    containing extra entries. For right now this will error out at 
    generate_event_list (hopefully) or at the extraction of btimes.
    
    Finally, trials_info and events are written to the RS directory as
        trials_info
        events
    """
    if not force and os.path.exists(os.path.join(rs.full_path, 'events')):
        return    
    
    # get events
    bcld = bcontrol.Bcontrol_Loader_By_Dir(rs.full_path)
    bcld.load()
    peh = bcld.data['peh']
    TI = bcontrol.demung_trials_info(bcld)
    
    # Generate event list ... why would TI_start_idx every be nonzero
    # for purely behavioral stuff? Possibly if the trials were regenerated
    # before starting the protocol ... error here and fix that edge case then.
    # That is the purpose of `extra_peh_offset` in write_b2n_sync_file
    rec_l = bcontrol.generate_event_list(peh, bcld.data['TRIALS_INFO'], 
        TI_start_idx=0, 
        error_check_last_trial=bcld.data['CONSTS']['FUTURE_TRIAL'])
    events = pandas.DataFrame.from_records(rec_l)

    # Optionally convert to neural time base
    if convert_to_ntime:
        # Load the sync file (generating if necessary)
        fname = os.path.join(rs.full_path, 'SYNC_B2N')        
        if not os.path.exists(fname):
            write_b2n_sync_file(rs, known_trial_numbers=known_trial_numbers)        
        sync_b2n = np.loadtxt(fname, dtype=np.float)
        
        # Apply the sync    
        events['time'] = np.polyval(sync_b2n, events.time)

    # Reindex TRIALS_INFO
    TI.set_index('trial_number', inplace=True)
    TI.index.name = 'trial_number'
    
    # Insert time ...
    # This is commented out for now because there are two tricky parts
    # 1) Deciding whether to use start time or stimulus time
    # 2) Accounting for possibility of extra_peh_offset
    # 3) Deciding how to calculate ... extract times from peh using
    #    sync? Use TRIAL_NUMBERS and TIMESTAMPS?
    #TI.insert(0, 'time', np.nan)
    #TI['time'][bskip:bskip+len(ntimes)] = ntimes
    
    # Output to RS
    events.to_csv(os.path.join(rs.full_path, 'events'), index=False, 
        header=False, sep='\t')
    TI.to_csv(os.path.join(rs.full_path, 'trials_info'), index=True, 
        header=True)        

def add_behavioral_trial_numbers(rs, bskip=1, copy_corr_files=True):
    """Syncs trial numbers and adds behavioral to Segment info field.
    
    There is a hacky parameter that skips the first `bskip` behavioral
    trials. There are two common use cases:
    
    * bskip = 1. Tested a lot. Basically, the first trial is often not
        recorded in the neural data. So don't include it in the syncer.
        If it was recorded, there's a try/except loop that is specific
        to the first neural trial, and assumes it's behavioral trial 0.
    
    * bskip > 1. This is for the case where you've removed a lot of
        trials from TIMESTAMPS manually. It won't be able to sync. So
        choose bskip to be about 10 trials or so less than the number you
        removed. Then it should be able to sync and find all of the neural
        trials that belong to the remaining behavioral trials. If it can't
        find a neural trial, then an error occurs (actually it's labelled
        -99 and a warning is printed). That's why you want to choose bskip
        to be 10 less than the actual number that were removed.
    
    If you removed TIMESTAMPS from the end, there should be no problem.
    The errors are worse when the behavioral trial can't be found because
    you skipped it

    You can also run this without even an existing database. (Though note
    this will create an empty db.) That would be useful for syncing things
    up before inserting a bunch of spurious timestamps into the db.

    Return:
        bs, a BehavingSyncer object
    """
    # fake a data structure that the syncer needs
    Kls = collections.namedtuple('kls', 'audio_onsets')
    
    # Load bcontrol data from matfile (will also validate)
    bcld = bcontrol.Bcontrol_Loader_By_Dir(rs.full_path, skip_trial_set=[])
    bcld.load()
    
    # Get onset times. For behavioral, it is in seconds and we skip the
    # first trial which often occurs before recording started. We correct
    # for this in the loop below when we add one to the index into
    # TRIALS_INFO.
    # Neural timestamps are in samples.
    behavioral_timestamps = Kls(audio_onsets=bcld.data['onsets'][bskip:])
    neural_timestamps = Kls(audio_onsets=rs.read_timestamps())
    
    # Sync. Will write CORR files to disk.
    # Also produces bs.map_n_to_b_masked and vice versa for trial mapping
    bs = DataSession.BehavingSyncer()
    bs.sync(behavioral_timestamps, neural_timestamps, force_run=True)

    # move the CORR files generated by bs
    if copy_corr_files and os.path.exists('CORR_NtoB'):
        shutil.copyfile('CORR_NtoB', os.path.join(rs.full_path, 'CORR_NtoB'))
    if copy_corr_files and os.path.exists('CORR_BtoN'):
        shutil.copyfile('CORR_BtoN', os.path.join(rs.full_path, 'CORR_BtoN'))
    
    # Put trial numbers into OE db
    session = rs.get_OE_session()
    seglist = session.query(OE.Segment).all()
    for seg in seglist:
        # Each segment in the db is named "Segment %d", corresponding to the
        # ordinal TIMESTAMP, which means neural trial time.
        n_trial = int(re.search('Segment (\d+)', seg.name).group(1))
        
        # Convert to behavioral trial number
        # We use the 'trial_number' field of TRIALS_INFO
        # IE the original Matlab numbering of the trial
        # Here we correct for the dropped first trial.
        try:
            b_trial = bcld.data['TRIALS_INFO']['TRIAL_NUMBER'][\
                bs.map_n_to_b_masked[n_trial] + bskip]
        except IndexError:
            # masked trial
            if n_trial == 0:
                if bskip != 1:
                    print "warning: can't find behavior trial for n_trial==0, this shouldn't happen with this bskip"
                #print "WARNING: Assuming this is the dropped first trial"
                b_trial = bcld.data['TRIALS_INFO']['TRIAL_NUMBER'][0]
            else:
                print "WARNING: can't find trial"
                b_trial = -99
        
        # Store behavioral trial number in the info field
        seg.info = '%d' % b_trial
        seg.save()

    # Pretty sure there is a missing session.commit() here
    # Oh well, will probably happen when the session variable is destroyed ...
    # Also, pretty sure the seg.save() is superfluous
    # This has been fixed in the newer version of this method

    return bs
    
    

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


# other utility wrapper functions
def guess_session_type(rs):
    """Guesses whether session is behaving or passive.
    
    If 0 or 1 timestamps, returns 'unknown'.
    If median time between timestamps is > 1s, returns 'behaving'
    If median time is < 1s, returns 'WN100ms'.
    """
    try:
        tss = rs.read_timestamps()
    except IOError:
        print "warning: no timestamps exist, cannot guess session type"
        return "unknown"
    
    if len(tss) < 2:
        return "unknown"
    elif np.median(np.diff(tss)) > 30000:
        return 'behaving'
    else:
        return 'WN100ms'

def generate_speakercal_timestamps(bout_onsets, n_tones=200, inter_tone=6000,
    tones_per_package=10, initial_offset=3359, package_offset=-165):
    """Helper function to convert from onsets to tone times in speakercal.
    
    Given a list of speakercal run onset times, this calculates when the
    tones were actually played within each run.
    
    All times are in samples.
    bout_onsets : list of start times of each run, detected from digital
        onset  
    inter_tone : nominal time between tones
    n_tones : number of tones per bout
    tones_per_package : number of tones in each audio waveform in bcontrol
    initial_offset : time between bout_onset and actual beginning of first
        tone. The default is 3359, which includes the following:
            3000 : wait state at beginning of trial
            159 : time taken to send the digital word
            200 : unknown
    package_offset : There is an additional offset for each package, but not
        within the package. I assume this is due to the miscalculation of
        stimulus time within speakercal. The default is negative, reflecting
        that each subsequent package occurs 165 samples EARLIER than expected.
        Note that this value dwarfs what would be expected by the .99663
        dilation (about 20 samples), and so therefore includes it.
    """
    
    res = []
    for bout_onset in bout_onsets:
        # Length n_tones : offset introduced by the package
        package_offsets = package_offset * np.array(
            [n_tone / tones_per_package for n_tone in range(n_tones)])
        
        # Length n_tones : offset introduced by the ITI
        tone_offsets = np.arange(n_tones, dtype=np.int) * inter_tone
        
        this_bout = np.rint(
            bout_onset + initial_offset + package_offsets + tone_offsets).\
            astype(np.int)
        
        res.append(this_bout)
        
    
    return np.concatenate(res)
