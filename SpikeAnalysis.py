"""Methods to process spike data and LBPB behavioral data

TODO: object that knows about sdf and tdf from a single session
This is necessary to, for instance, count # of correct trials during PSTHing.
Should probably sanitize the sound names, outcomes, etc into strings during loading.

Separate plotting code from computation code

Create consistent interface for plotting average response over units (or tetrodes),
and plotting just one.
"""
import numpy as np
import pandas
import glob
import os.path
import bcontrol
import myutils
import matplotlib.pyplot as plt
import scipy.stats

def get_stim_idxs_and_names():
    """Returns list of sound names and indexes.
    
    sound_names, groups = get_stim_idxs_and_names()
    In [261]: sound_names
    Out[261]: [u'le_hi', u'ri_hi', u'le_lo', u'ri_lo']

    In [262]: groups
    Out[262]: [array([5, 9]), array([ 6, 10]), array([ 7, 11]), array([ 8, 12])]
    
    PB is always first and LB is always second
    """
    consts = bcontrol.LBPB_constants()
    names, idx, groupnames = consts.comparisons(comp='sound')
    
    # this is the list of pairs of sound idxs
    groups = map(lambda group: np.array(group).flatten(), idx)
    
    sound_names = [groupname[0][:5].replace('_', '') 
        for groupname in groupnames]
    
    return sound_names, groups

def integrate_sdf_and_tdf(sdf, tdf, F_SAMP=30e3, n_bins=75, t_start=-.25,
    t_stop=.5, split_on_unit=True, include_trials='hits'):
    """Use trial information to fold spike information into trial structure"""
    t_vals = np.linspace(t_start, t_stop, n_bins + 1)
    
    assert include_trials == 'hits', "for now"
    
    # Keep only hits
    tdf_hits = tdf[(tdf.outcome == 1) & (tdf.nonrandom == 0)]
    
    sound_names, groups = get_stim_idxs_and_names()

    data = pandas.DataFrame(columns=['block', 'counts', 'sound', 
        'trials', 'bin'])

    # Bin
    for sound_name, group in zip(sound_names, groups):
        # First PB, then LB
        for n, block in enumerate(['PB', 'LB']):
            trial_numbers = tdf_hits[tdf_hits.stim_number == group[n]].index
            spike_subset = sdf[sdf.trial.isin(trial_numbers)]
            
            if not split_on_unit:
                spike_times = spike_subset.adj_time
                counts, junk = np.histogram(spike_times / F_SAMP, 
                    bins=t_vals)
                
                this_frame = pandas.DataFrame({
                    'counts': counts,
                    'block': [block]*len(counts),
                    'sound': [sound_name]*len(counts),
                    'bin': range(len(counts)),
                    'trials': [len(trial_numbers)]*len(counts)})    
                
                data = data.append(this_frame, ignore_index=True)
            else:
                for unit_num in np.unique(np.asarray(sdf.unit)):
                    spike_times = \
                        spike_subset[spike_subset.unit == unit_num].adj_time
                    counts, junk = np.histogram(spike_times / F_SAMP, 
                        bins=t_vals)
                    
                    this_frame = pandas.DataFrame({
                        'counts': counts,
                        'block': [block]*len(counts),
                        'sound': [sound_name]*len(counts),
                        'bin': range(len(counts)),
                        'trials': [len(trial_numbers)]*len(counts),
                        'unit': [unit_num]*len(counts)})    
                    
                    data = data.append(this_frame, ignore_index=True)                    
    
    return data

class SpikeServer:
    def __init__(self, base_dir, filename_filter=None, 
        SU_list_filename=None):
        """Expects base_dir to be a directory containing files like:
        
        session_name1_spikes
        session_name1_trials
        session_name2_spikes
        etc.
        
        filename_filter: a string. Only sessions including this string will
            be analyzed.
        """
        self.base_dir = base_dir
        self.filename_filter = filename_filter
        self.refresh_files()
    
    def refresh_files(self):
        """Rechecks filenames from disk for spike and trial files"""
        # Find spike and trial files in base directory
        self.sdf_list = sorted(glob.glob(os.path.join(self.base_dir, 
            '*_spikes')))
        self.tdf_list = sorted(glob.glob(os.path.join(self.base_dir, 
            '*_trials')))
        
        # Optionally filter
        if self.filename_filter is not None:
            self.sdf_list = filter(lambda s: self.filename_filter in s, 
                self.sdf_list)
            self.tdf_list = filter(lambda s: self.filename_filter in s, 
                self.tdf_list)
        
        # convert to array
        self.sdf_list = np.asarray(self.sdf_list)
        self.tdf_list = np.asarray(self.tdf_list)
        
        
        # Extract session names and error check
        self.session_names = np.array(map(lambda s: os.path.split(s)[1][:-7], 
            self.sdf_list))

        temp = np.asarray(map(lambda s: os.path.split(s)[1][:-7], 
            self.tdf_list))
        assert np.all(self.session_names == temp)
        
        if len(self.session_names) == 0:
            print "warning: no data found"
    
    def read_flat_spikes_and_trials(self, session='all', include_trials='hits',
        unit_filter=None, stim_number_filter=None):
        """Returns flat data frame of spike times from specified trials.
        
        unit_filter: dict session_name -> array of unit numbers to keep.
        stim_number_filter : list of stimulus numbers to keep
        
        TODO: implement tetrode filter, currently it's assumed that bad
        tetrdoes have already been stripped from underlyng files.
        
        Returns DataFrame in this format:        
        <class 'pandas.core.frame.DataFrame'>
        Int64Index: 800464 entries, 0 to 800463
        Data columns:
        adj_time       800464  non-null values
        session        800464  non-null values
        spike_time     800464  non-null values
        stim_number    800464  non-null values
        tetrode        800464  non-null values
        trial          800464  non-null values
        unit           800464  non-null values
        dtypes: int64(6), object(1)
        
        This allows, for instance:
        fsd = ss.flat_spike_data()
        for (session, unit), df in fsd.groupby(['session', 'unit']):
            print session
            print unit
            print df
        
        Or:
        for stim_number, df in fsd.groupby(['stim_number']):
            figure(); hist(np.asarray(df.adj_time))
        """
        flat_spike_data = pandas.DataFrame()
        
        # which sessions to get?
        if session == 'all':
            session_mask = np.ones(self.session_names.shape, dtype=np.bool)
        else:
            session_mask = (self.session_names == session)
            assert np.any(session_mask), "can't find session %s" % session
        
        # iterate through sessions
        iter_obj = zip(self.session_names[session_mask],
            self.sdf_list[session_mask], self.tdf_list[session_mask])
        for session_name, sdf_fn, tdf_fn in iter_obj:
            sdf = pandas.load(sdf_fn)
            tdf = pandas.load(tdf_fn)
            found_units = np.unique(np.asarray(sdf.unit))
        
            # here is where unit filtering goes
            if unit_filter is not None:
                units_to_keep = unit_filter[session_name]
                if np.any(~np.in1d(units_to_keep, found_units)):
                    print "warning: could not find all requested units in " + \
                        "session %s" % session_name
                sdf = sdf[sdf.unit.isin(units_to_keep)]
            
            # discard bad trials
            if include_trials == 'hits':
                tdf = tdf[(tdf.outcome == 1) & (tdf.nonrandom == 0)]
            else:
                raise "only hits supported for now"
            
            # stimulus filtering
            if stim_number_filter is not None:
                tdf = tdf[tdf.stim_number.isin(stim_number_filter)]
            
            # retain only spikes from good trials
            sdf = sdf[sdf.trial.isin(tdf.index)]
            
            # insert key to this session
            sdf.insert(loc=0, column='session', value=[session_name]*len(sdf))
            
            # add in stimulus number
            sdf = sdf.join(tdf.stim_number, on='trial')
            
            flat_spike_data = flat_spike_data.append(sdf, ignore_index=True)
        
        return flat_spike_data
    
    def count_trials_by_type(self, session=None, include_trials='hits', 
        **kwargs):
        return len(self.list_trials_by_type(session, include_trials, **kwargs))
    
    def list_trials_by_type(self, session=None, include_trials='hits',
        **kwargs):
        idx = np.nonzero(self.session_names == session)[0].item()
        tdf = pandas.load(self.tdf_list[idx])
        if include_trials == 'hits':
            tdf = tdf[(tdf.outcome == 1) & (tdf.nonrandom == 0)]
        else:
            raise "only hits supported for now"

        replace_stim_numbers_with_names(tdf)
        
        mask = myutils.pick_mask(tdf, **kwargs)        
        return np.asarray(tdf.index[mask], dtype=np.int)



def bin_flat_spike_data(fsd, trial_counter=None, F_SAMP=30e3, n_bins=75, 
    t_start=-.25, t_stop=.5):
    """Bins in time over trials, removing trial numbers and spike times.
    
    The following columns MUST exist:
        adj_time : these are binned in the histogram
        trial : these are used to determine how many trials occurred
    They will be stripped from the returned data, in addition to 'spike_time'
    if that is a column.
    
    The following columns will be added:
        counts : number of spikes in that bin
        trials : number of trials during which spikes could have occurred
    
    So you might get back something like this:
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 57600 entries, 0 to 57599
    Data columns:
    session        57600  non-null values
    stim_number    57600  non-null values
    tetrode        57600  non-null values
    unit           57600  non-null values
    counts         57600  non-null values
    trials         57600  non-null values
    time           57600  non-null values
    dtypes: object(7)
    """
    # how to hist
    t_vals = np.linspace(t_start, t_stop, n_bins + 1)
    
    # we will group over all columns except the ones to be removed
    cols = fsd.columns.tolist()
    cols.remove('adj_time')
    cols.remove('trial')
    if 'spike_time' in cols:
        cols.remove('spike_time')
    
    # iterate over the groups and bin each one
    rec_l = []
    for key, df in fsd.groupby(cols):
        # histogramming
        counts, junk = np.histogram(df.adj_time / F_SAMP, bins=t_vals)
        
        # count trials
        n_trials = None
        if trial_counter is None:
            print "warnign: no trial info provided, using length"
            n_trials = len(np.unique(df.trial))
        else:
            try:
                block_name = np.unique(np.asarray(df.block))
                sound_name = np.unique(np.asarray(df.sound))
                session = np.unique(np.asarray(df.session))
            except AttributeError:
                print "warning: cannot get sound/block, using length"
                n_trials = len(np.unique(df.trial))            
        if n_trials is None:
            if len(block_name) > 1 or len(sound_name) > 1 or len(session) > 1:
                print "warning: non-unique sound/block/session, using length"
                n_trials = len(np.unique(df.trial))
            else:
                n_trials = trial_counter(session=session, block=block_name, 
                    sound=sound_name)
        
        if n_trials < len(np.unique(df.trial)):
            raise ValueError("counted more trials than exist")
        
        # Add in the keyed info (session etc), plus n_counts, n_trials, and bin
        this_frame = [list(key) + [count, n_trials, t_val] 
            for count, t_val in zip(counts, t_vals[:-1])]
        
        # append to growing list
        rec_l += this_frame
    
    # convert to new data frame, using same keyed columns plus our new ones
    cols = cols + ['counts', 'trials', 'time']
    newdf = pandas.DataFrame(rec_l, columns=cols)
    return newdf

def replace_stim_numbers_with_names(df, strip_old_column=True):
    """Converts column called stim_number into columns called sound and block.
    
    Changes df in place.
    """
    sound_names, groups = get_stim_idxs_and_names()
    fmt = '|S%d' % max([len(s) for s in sound_names])
    name_col = np.array(['']*len(df), dtype=fmt)
    block_col = np.array(['']*len(df), dtype='|S2')

    for sound_name, group in zip(sound_names, groups):
        block_col[df.stim_number == group[0]] = 'PB'
        block_col[df.stim_number == group[1]] = 'LB'        
        name_col[df.stim_number.isin(group)] = sound_name
    
    df['block'] = block_col
    df['sound'] = name_col
    
    if strip_old_column:
        df.pop('stim_number')

def create_unique_neural_id(df, split_on='unit', column='nid'):
    """Adds column with a unique integer label for tetrode or unit * session.
    """
    df[column] = -np.ones(len(df), dtype=np.int)
    g = df.groupby(['session', split_on])
    for n, (key, idxs) in enumerate(g.groups.items()):
        df[column][idxs] = n

def plot_psths_by_sound_from_flat(fdf, trial_lister=None, fig=None, ymax=1.0):
    """Plots PSTHs by sound from flat frame, to allow rasters.
    
    
    """
    # get session name
    session_l = np.unique(np.asarray(fdf.session))
    if len(session_l) != 1:
        print "error: must be exactly one session!"
        1/0
    session = session_l[0]
    
    if fig is None:
        fig = plt.figure()

    # iterate over sounds and plot each
    g1 = fdf.groupby(['sound', 'block'])    
    for n, sound_name in enumerate(['lehi', 'rihi', 'lelo', 'rilo']):
        # get axis
        try:
            ax = fig.axes[n]
        except IndexError:
            ax = fig.add_subplot(2, 2, n + 1)
        
        # iterate over blocks
        for block_name in ['LB', 'PB']:
            # get spikes form this sound * block
            x = fdf.ix[g1.groups[sound_name, block_name]]
        
            # group those spikes by the trial from which they came
            g2 = x.groupby('trial')
            trial_list = trial_lister(session=session, sound=sound_name, 
                block=block_name)
            
            # grab spikes by trial
            folded_spikes = []
            for trial_number in trial_list:
                try:
                    spike_idxs = g2.groups[trial_number]
                except KeyError:
                    spike_idxs = np.array([])
                folded_spikes.append(
                    np.asarray(x.ix[spike_idxs]['adj_time']) / 30000.0)
            
            # error check
            n_empty_trials = sum([len(s) == 0 for s in folded_spikes])
            assert (n_empty_trials + len(np.unique(x.trial))) == len(trial_list)
            
            old_xlim = ax.get_xlim()
            if block_name == 'LB':
                myutils.plot_rasters(folded_spikes, ax=ax, full_range=0.25,
                    y_offset=-.5, plot_kwargs={'color': 'b'})
            if block_name == 'PB':
                myutils.plot_rasters(folded_spikes, ax=ax, y_offset=-0.25,
                    full_range=0.25, plot_kwargs={'color': 'r'})
            ax.set_xlim((-.25, .5))
            

    
    

def plot_psths_by_sound(df, plot_difference=True, split_on=None,
    mark_significance=False, plot_errorbars=True, p_adj_meth=None):
    """Plots PSTHs for each sound, for a single unit or average across multiple.
    
    df : DataFrame containing binned data with following columns
        :split_on: - Tell me how to group the data
            eg 'nid', or ['session', 'unit'], or None (single trace)
        time - time in sec
        counts - number of spikes in that bin
        trials - number of trials included in that bin
        sound, block - what the stimulus was
    
    Run bin_flat_spike_data and replace_stim_numbers_with_names to produce
    these columns.
    """
    # Create figure
    f = plt.figure(figsize=(9,9))
    
    # Pivot data to allow sound/block access.
    if split_on is None:
        # Aggregate all data into a single trace
        pdata = df.pivot_table(rows=['time'], cols=['sound', 'block'],
            values=['counts', 'trials'], aggfunc=np.sum)
    else:
        # Split according to request
        pdata = df.pivot_table(rows=['time'], cols=['sound', 'block', split_on],
            values=['counts', 'trials'], aggfunc=np.sum)
    
    # Double-check we didn't lose data
    if len(df) != pdata['counts'].shape[0] * pdata['counts'].shape[1]:
        print "warning: refolded data doesn't match shape"
        print "avoid this warning by passing a single trace, or specifying merge column"
    
    # Iterate over sounds (one axis per sound)
    for n, sound_name in enumerate(['lehi', 'rihi', 'lelo', 'rilo']):
        LB_counts = pdata['counts'][sound_name]['LB'].values.astype(np.int)
        LB_trials = pdata['trials'][sound_name]['LB'].values.astype(np.int)
        PB_counts = pdata['counts'][sound_name]['PB'].values.astype(np.int)
        PB_trials = pdata['trials'][sound_name]['PB'].values.astype(np.int)
        
        # get time vector and check for consistency
        # technically could be in arbitrary order, in which case will error
        times = pdata['counts'][sound_name]['LB'].index.values.astype(np.float)
        assert (times == pdata['trials'][sound_name]['LB'].index.values).all()
        assert (times == pdata['counts'][sound_name]['PB'].index.values).all()
        assert (times == pdata['trials'][sound_name]['PB'].index.values).all()
        
        # Create axis for this plot and plot means with errorbars
        ax = f.add_subplot(2, 2, n + 1)
        myutils.plot_mean_trace(ax=ax, x=times, 
            data=LB_counts / LB_trials.astype(np.float), 
            label='LB', color='b', axis=1, errorbar=True)
        myutils.plot_mean_trace(ax=ax, x=times, 
            data=PB_counts / PB_trials.astype(np.float), 
            label='PB', color='r', axis=1, errorbar=True)
        
        if plot_difference:
            myutils.plot_mean_trace(ax=ax, x=times,
                data=(
                    LB_counts / LB_trials.astype(np.float) -
                    PB_counts / PB_trials.astype(np.float)),
                label='diff', color='m', axis=1, errorbar=True)
            ax.plot(times, np.zeros_like(times), 'k')
        
        if mark_significance:
            LB_rate = LB_counts / LB_trials.astype(np.float)
            PB_rate = PB_counts / PB_trials.astype(np.float)
            p_vals = scipy.stats.ttest_rel(LB_rate.transpose(), 
                PB_rate.transpose())[1]
            
            if p_adj_meth is not None:
                p_vals = myutils.r_adj_pval(p_vals, meth=p_adj_meth)
            
            pp = np.where(p_vals < .05)[0]
            plt.plot(times[pp], np.zeros_like(pp), 'k*')
            
            pp = np.where(p_vals < .01)[0]
            plt.plot(times[pp], np.zeros_like(pp), 'ko',
                markerfacecolor='w')
        
        ax.set_title(sound_name)
        plt.legend()

    plt.show()    
    return f