"""Methods to process spike data and LBPB behavioral data

TODO: object that knows about sdf and tdf from a single session
This is necessary to, for instance, count # of correct trials during PSTHing.
Should probably sanitize the sound names, outcomes, etc into strings during loading.

Separate plotting code from computation code

Create consistent interface for plotting average response over units (or tetrodes),
and plotting just one.

There are three formats that are useful:
    flat (one row per spike)
    binned (lowest level is bins * variable # of trials)
    binned and averaged (lowest level is simply bins)

It would also be useful to have (sound, block) iterators over those formats,
which the plotting functions would use.

Also, a containing object to remember time-parameters like f_samp, binwidth
"""
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import pandas
import glob
import os.path
from . import bcontrol
from . import myutils
import matplotlib.pyplot as plt
import scipy.stats
import rpy2.robjects as robjects
r = robjects.r


def r_poisson_test(count1, count2, trials1=1, trials2=1):
    if trials1 == 0 or trials2 == 0:
        return np.nan
    r_call = "poisson.test(c(%d,%d), c(%d,%d))$p.value" % (
        count1, count2, trials1, trials2)
    p = r(r_call)[0]
    return p

def r_poisson_test_array(count1, count2, trials1, trials2):
    return np.array([r_poisson_test(c1, c2, v1, v2) 
        for c1, c2, v1, v2 in zip(
        np.array(count1).flatten(), 
        np.array(count2).flatten(), 
        np.array(trials1).flatten(),
        np.array(trials2).flatten())]).reshape(count1.shape)


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
        SU_list_filename=None, f_samp=30e3, t_start=-.25, t_stop=.5,
        bins=75):
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
        self.f_samp = f_samp
        self.t_start = t_start
        self.t_stop = t_stop
        self.bins = bins
    
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
            print("warning: no data found")
    
    def read_flat_spikes_and_trials(self, session='all', include_trials='hits',
        unit_filter=None, stim_number_filter=None):
        """Returns flat data frame of spike times from specified trials.
        
        Also memoizes the result as self._fsd.
        
        unit_filter: dict session_name -> array of unit numbers to keep.
        stim_number_filter : list of stimulus numbers to keep
        
        TODO: implement tetrode filter, currently it's assumed that bad
        tetrdoes have already been stripped from underlyng files.
        
        Filtering:
        * Specify session to load only from one session, or all to load all
        * Specify unit_filter as a dict session: [list of units] to keep
          only those units. Sessions not in the unit filter are skipped.
        
        TODO: make this filtering parameter a dataframe. So a list of
        session, tetrode/unit records.
        
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
        if hasattr(self, '_fsd'):
            return self._fsd
        
        flat_spike_data = pandas.DataFrame()
        
        # which sessions to get?
        if session == 'all':
            session_mask = np.ones(self.session_names.shape, dtype=np.bool)
        else:
            session_mask = (self.session_names == session)
            assert np.any(session_mask), "can't find session %s" % session
        
        # iterate through sessions
        df_list = []
        iter_obj = zip(self.session_names[session_mask],
            self.sdf_list[session_mask], self.tdf_list[session_mask])
        for session_name, sdf_fn, tdf_fn in iter_obj:
            if unit_filter is not None and session_name not in unit_filter:
                continue
            
            sdf = pandas.load(sdf_fn)
            tdf = pandas.load(tdf_fn)
            found_units = np.unique(np.asarray(sdf.unit))
        
            # here is where unit filtering goes
            if unit_filter is not None:
                units_to_keep = unit_filter[session_name]
                if np.any(~np.in1d(units_to_keep, found_units)):
                    print("warning: could not find all requested units in " + \
                        "session %s" % session_name)
                sdf = sdf[sdf.unit.isin(units_to_keep)]
            
            # discard bad trials
            if include_trials == 'hits':
                tdf = tdf[(tdf.outcome == 1) & (tdf.nonrandom == 0)]
            elif include_trials == 'non-hits':
                tdf = tdf[(tdf.outcome != 1) & (tdf.nonrandom == 0)]
            elif include_trials == 'all random':
                tdf = tdf[tdf.nonrandom == 0]
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
            
            df_list.append(sdf)
            #flat_spike_data = flat_spike_data.append(sdf, ignore_index=True)
        
        self._fsd = pandas.concat(df_list, ignore_index=True)
        return self._fsd
    
    def count_trials_by_type(self, session=None, include_trials='hits', 
        **kwargs):
        return len(self.list_trials_by_type(session, include_trials, **kwargs))
    
    def list_trials_by_type(self, session=None, include_trials='hits',
        **kwargs):
        idx = np.nonzero(self.session_names == session)[0].item()
        tdf = pandas.load(self.tdf_list[idx])
        if include_trials == 'hits':
            tdf = tdf[(tdf.outcome == 1) & (tdf.nonrandom == 0)]
        elif include_trials == 'non-hits':
            tdf = tdf[(tdf.outcome != 1) & (tdf.nonrandom == 0)]
        elif include_trials == 'all random':
            tdf = tdf[tdf.nonrandom == 0]
        else:
            raise "only hits, non-hits, and all random supported for now"

        replace_stim_numbers_with_names(tdf)
        
        mask = myutils.pick_mask(tdf, **kwargs)        
        return np.asarray(tdf.index[mask], dtype=np.int)
    
    def pick_trials(self, **kwargs):
        """Chooses sessions*trials based on filter.
        
        First loads all trials from all sessions.
        Then, for each kwarg, chooses trials that match kwarg.
        
        Unless you override, nonrandom == [0] and outcome == [1]
        
        Example:
        pick_trials(session=['session1'], outcome=['hit'], stim_number=[3,5])
        
        This returns a masked version of all_trials including only the matches.
        The index is ['session', 'trial']
        """
        if 'outcome' not in kwargs:
            kwargs['outcome'] = [1]
        if 'nonrandom' not in kwargs:
            kwargs['nonrandom'] = [0]
        
        # uses memoized property self.all_trials
        at = self.all_trials.reset_index()
        mask = np.ones(len(at), dtype=bool)
        for key, vals in kwargs.items():
            mask &= at[key].isin(vals)
        
        
        return at[mask].set_index(['session', 'trial'])
        
    
    def get_binned_spikes_by_trial(self, split_on, split_on_filter=None,
        f_samp=30e3, t_start=-.25, t_stop=.5, bins=75, include_trials='hits'):
        """Returns binned data separately for each trial.
        
        There is a variable number of columns bin%d, depending on the number
        you request.
        
        Format:
        <class 'pandas.core.frame.DataFrame'>
        Int64Index: 23202 entries, 0 to 23201
        Data columns:
        session    23202  non-null values
        tetrode    23202  non-null values
        sound      23202  non-null values
        block      23202  non-null values
        trial      23202  non-null values
        bin0       23202  non-null values
        dtypes: int64(3), object(3)
        """

        fsd = self.read_flat_spikes_and_trials(stim_number_filter=range(5,13),
            include_trials=include_trials)
        replace_stim_numbers_with_names(fsd)
        
        g = fsd.groupby(split_on)
        
        #df = pandas.DataFrame()
        dfs = []
        for key, val in g:
            if split_on_filter is not None and key not in split_on_filter:                
                continue
                
        
            for sound_name in ['lehi', 'rihi', 'lelo', 'rilo']:
                for block_name in ['LB', 'PB']:
                    # subframe
                    subdf = val[(val.sound == sound_name) & 
                        (val.block == block_name)]
                    
                    # get session name
                    session_l = np.unique(np.asarray(subdf.session))
                    if len(session_l) == 0:
                        continue
                    elif len(session_l) > 1:
                        raise "Non-unique sessions, somehow"
                    else:
                        session = session_l[0]

                    trial_list = self.list_trials_by_type(session=session,
                        sound=sound_name, block=block_name, 
                        include_trials=include_trials)
            
                    counts, times = myutils.times2bins(
                        fold(subdf, trial_list),
                        f_samp=f_samp, t_start=t_start, t_stop=t_stop, bins=bins, 
                        return_t=True)
                    
                    this_frame = [list(key) + [sound_name, block_name, trial] 
                        + list(count)
                        for count, trial in zip(counts, trial_list)]
                    
                    dfs.append(pandas.DataFrame(this_frame,
                        columns=(split_on + ['sound', 'block', 'trial'] +
                        ['bin%d' % n for n in range(counts.shape[1])])))#,
                    #    ignore_index=True)
            
        return pandas.concat(dfs, ignore_index=True)

    def get_binned_spikes3(self, spike_filter=None, trial_filter=None):
        """Generic binning function operating on self._fsd
        
        spike_filter : dataframe describing how to split fsd
            The columns are the hierarchy to split on:
                eg ['session', 'unit']
            The items are the ones to include.
            If no items, then everything is included.
            If None, then bin over everything except 'adj_time' or 'spike_time'
            
            Here we delineate every combination because it's not separable
            over session and unit (usually).
        
        trial_filter :
            How to do this filtering?
            For instance: hits only, stim_numbers 1-12, expressed as dicts
            In this case I wouldn't want to enumerate every combination
            because I can just take intersection over stim numbers and outcome.
            
            Although note we might want to combine errors and wrongports,
            for instance.
            
            It's implicit that we want to do this for each session in
            spike_filter.
        

        First the spikes are grouped over the columns in spike_filter.
        For each group, the trials are grouped over the columns in trial_filter.
        This cross-result is histogrammed.
        All results are concatenated and returned.

        The actual histogramming is done by myutils.times2bins using
        self.f_samp, t_start, t_stop, bins
        
        
        """
        input = self._fsd
        
        # default, use all columns and include all data
        if spike_filter is None:
            col_list = input.columns.tolist()
            remove_cols = ['adj_time', 'spike_time']
            for col in remove_cols:
                if col in col_list:
                    col_list.remove(col)
            spike_filter = pandas.DataFrame(columns=col_list)
        
        # Choose data from `input` by defining the following variables:
        #   `keylist` : a list of keys to include, each separately binned
        #   `grouped_data` : a dict from each key in keylist, to the data
        #   `keynames` : what to call each entry of the key in the result
        if len(spike_filter) == 0:
            # use all data
            keynames = spike_filter.columns.tolist()
            keylist = [tuple([myutils.only_one(input[col])
                for col in keynames])]
            val = input
            grouped_data = {keylist[0]: val}            
        elif len(spike_filter) == 1:
            # Optimized for the case of selecting a single unit
            d = {}
            for col in spike_filter:
                d[col] = spike_filter[col][0]            
            mask = myutils.pick_mask(input, **d)
            
            keylist = spike_filter.to_records(index=False) # length 1
            keynames = spike_filter.columns.tolist()        
            grouped_data = {keylist[0] : input.ix[mask]}
        else:
            # standard case
            g = input.groupby(spike_filter.columns.tolist())
            grouped_data = g.groups
            keylist = spike_filter.to_records(index=False)
            keynames = spike_filter.columns.tolist()
        
        # Now group the trials
        att = self.all_trials.reset_index().set_index(['session', 'trial'], drop=False)
        g2 = att.groupby(trial_filter.columns.tolist())
        #g2 = self.all_trials.groupby(trial_filter.columns.tolist())
        
        
        # Now iterate through the keys in keylist and the corresponding values
        # in grouped_data.
        rec_l = []    
        for key in keylist:
            # Take the data from this group
            subdf = grouped_data[key]
            
            for g2k, g2v in g2:
                # count trials of this type from this session
                session = myutils.only_one(subdf.session)
                n_trials = len(g2v.ix[session])
                if n_trials == 0:
                    # for example if a possible combination never actually
                    # occurred
                    continue

                # Join the spikes on the columns of trial filter
                subsubdf = subdf.join(g2v[trial_filter.columns], 
                    on=['session', 'trial'], how='inner', rsuffix='rrr')
                
                # check for already-joined columns
                for col in trial_filter.columns:
                    if col+'rrr' in subsubdf.columns:
                        assert (subsubdf[col] == subsubdf[col+'rrr']).all()
                        subsubdf.pop(col + 'rrr')
            
                # histogramming
                counts, t_vals = myutils.times2bins(
                    np.asarray(subsubdf.adj_time), return_t=True, 
                    f_samp=self.f_samp, t_start=self.t_start,
                    t_stop=self.t_stop, bins=self.bins)
                
                # Add in the keyed info (session etc), plus 
                # n_counts, n_trials, and bin
                frame_label = list(key) + list(np.array([g2k]).flatten())
                this_frame = [frame_label +
                    [count, n_trials, t_val, bin] 
                    for bin, (count, t_val) in enumerate(zip(counts, t_vals))]
                
                # append to growing list
                rec_l += this_frame
        
        # convert to new data frame, using same keyed columns plus our new ones
        cols = keynames + trial_filter.columns.tolist() + [
            'counts', 'trials', 'time', 'bin']
        newdf = pandas.DataFrame(rec_l, columns=cols)
        
        # drop the combinations that never actually occurred (future trials)
        return newdf
    
    
    def read_all_trials(self):
        return pandas.concat(dict([(session_name, pandas.load(tdf)) 
            for session_name, tdf in zip(self.session_names, self.tdf_list)]))
    
    @property
    def all_trials(self):
        try:
            return self._all_trials
        except AttributeError:
            self._all_trials = self.read_all_trials()
            self._all_trials.index.names = ['session', 'trial']
            return self._all_trials

def bin_flat_spike_data2(fsd, trial_counter=None, F_SAMP=30e3, n_bins=75, 
    t_start=-.25, t_stop=.5, split_on=None, include_trials='hits',
    split_on_filter=None):
    """Bins in time over trials, splitting on split_on.
    
    fsd : a flat array of spike times, with replaced stimulus names
    split_on : REQUIRED, how to split fsd, eg ['session', 'unit']
    split_on_filter : list of keys to be included, after splitting
        if None, then everything is included
    
    It will be separately binned over sound.
    """
    
    if split_on is None:
        split_on = []
    
    # iterate over the groups and bin each one
    rec_l = []    
    for key, df in fsd.groupby(split_on):
        if split_on_filter is not None and key not in split_on_filter:
            continue
        
        for sound_name in ['lehi', 'rihi', 'lelo', 'rilo']:
            for block_name in ['LB', 'PB']:
                # subframe
                subdf = df[(df.sound == sound_name) & (df.block == block_name)]
                
                # get session name
                session_l = np.unique(np.asarray(df.session))
                assert len(session_l) == 1
                session = session_l[0]
                
                # histogramming
                counts, t_vals = myutils.times2bins(
                    np.asarray(subdf.adj_time), f_samp=F_SAMP, 
                    t_start=t_start, t_stop=t_stop, bins=n_bins,
                    return_t=True)
        
                # count trials
                n_trials = trial_counter(session=session, block=block_name, 
                    sound=sound_name, include_trials=include_trials)
                
                # comment this out, because user might request subset of
                # original trial set
                #if n_trials < len(np.unique(np.asarray(subdf.trial))):
                #    raise ValueError("counted more trials than exist")
        
                # Add in the keyed info (session etc), plus 
                # n_counts, n_trials, and bin
                this_frame = [list(key) + 
                    [sound_name, block_name, count, n_trials, t_val] 
                    for count, t_val in zip(counts, t_vals)]
                
                # append to growing list
                rec_l += this_frame
    
    # convert to new data frame, using same keyed columns plus our new ones
    cols = split_on + ['sound', 'block', 'counts', 'trials', 'time']
    newdf = pandas.DataFrame(rec_l, columns=cols)
    return newdf


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
            print("warnign: no trial info provided, using length")
            n_trials = len(np.unique(np.asarray(df.trial)))
        else:
            try:
                block_name = np.unique(np.asarray(df.block))
                sound_name = np.unique(np.asarray(df.sound))
                session = np.unique(np.asarray(df.session))
            except AttributeError:
                print("warning: cannot get sound/block, using length")
                n_trials = len(np.unique(np.asarray(df.trial)))
        if n_trials is None:
            if len(block_name) > 1 or len(sound_name) > 1 or len(session) > 1:
                print("warning: non-unique sound/block/session, using length")
                n_trials = len(np.unique(np.asarray(df.trial)))
            else:
                n_trials = trial_counter(session=session, block=block_name, 
                    sound=sound_name)
        
        if n_trials < len(np.unique(np.asarray(df.trial))):
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
    if 'sound' in df.columns:
        return
    
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


def plot_rasters_and_psth(fsd, ss, fig=None, ymax=0.5, xlim=None,
    only_rasters=False, split_on=None, split_on_filter=None,
    plot_difference=False, ms=4, n_bins=75, t_start=-.25, t_stop=.5,
    include_trials='hits'):
    """Plots rasters of a specific unit by sound, and optionally the PSTH.
    
    Easiest is to pass the whole flat spike data, then specify
    split_on and split_on_filter (as just one unit).
    
    A wrapper around plot_psths_by_sound_from_flat, bin_flat_spike_data2,
    and plot_psths_by_sound.
    """
    # Filter out single desired unit
    if split_on is not None:
        mask = np.ones(len(fsd), dtype=np.bool)
        for col, val in zip(split_on, split_on_filter):
            mask = mask & (fsd[col] == val)        
        fsd = fsd.ix[mask]
    
    # Create figure and optionally plot psth
    if fig is None:
        if only_rasters:
            fig = plt.figure()
        else:
            # Plot PSTHs, so first bin
            bsd = bin_flat_spike_data2(fsd, 
                trial_counter=ss.count_trials_by_type, 
                split_on=split_on, split_on_filter=None,
                include_trials=include_trials, n_bins=n_bins, 
                t_start=t_start, t_stop=t_stop)
            
            # Now plot and grab figure handle
            plot_psths_by_sound(bsd, split_on=split_on, 
                plot_difference=plot_difference)
            fig = plt.gcf()
    
    plot_psths_by_sound_from_flat(fsd, trial_lister=ss.list_trials_by_type,
        fig=fig, ymax=ymax, xlim=xlim, ms=ms)
    
    yl2 = max([ax.get_ylim()[1] for ax in fig.axes])
    
    for ax in fig.axes:
        ax.plot(ax.get_xlim(), [0, 0], 'k-')
        ax.set_ylim((-ymax, yl2))
    plt.show()

def plot_psths_by_sound_from_flat(fdf, trial_lister=None, fig=None, ymax=1.0,
    xlim=None, ms=4):
    """Plots rasters of a specific unit by sound, and optionally the PSTH.
    
    fdf : flat data frame, from SpikeSorter.read_flat_*
        You must first filter out the unit you want.
    also_plot_average : if True, will first plot the PSTH by calling
        plot_psths_by_sound, then will plot rasters into same figure.
    
    LB spikes are blue, PB spikes are red.
    """
    if xlim is None:
        xlim = (-.25, .5)
    
    # get session name
    session_l = np.unique(np.asarray(fdf.session))
    if len(session_l) != 1:
        print("error: must be exactly one session!")
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
            ax.set_title(sound_name)
        
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
            # comment this out, because might be requesting subset of
            # original trial set
            #assert (n_empty_trials + len(np.unique(np.asarray(x.trial)))) == len(trial_list)
            
            old_xlim = ax.get_xlim()
            if block_name == 'LB':
                myutils.plot_rasters(folded_spikes, ax=ax, full_range=ymax/2.,
                    y_offset=-ymax, plot_kwargs={'color': 'b', 'ms': ms})
            if block_name == 'PB':
                myutils.plot_rasters(folded_spikes, ax=ax, y_offset=-ymax/2.,
                    full_range=ymax/2., plot_kwargs={'color': 'r', 'ms': ms})
            ax.set_xlim(xlim)

def compare_rasters(bspikes1, bspikes2, meth='ttest', p_adj_meth=None,
    mag_meth='diff', fillval=None):
    """Compare two frames across replicates.
    
    bspikes1.shape = (N_features, N_trials1)
    bspikes2.shape = (N_features, N_trials2)
    
    Returns shape (N_features,), calculated as:
        mean_diff   : np.mean(A - B, axis=1)
        sqrt_diff   : np.mean(np.sqrt(A) - np.sqrt(B))
        
    """
    if mag_meth == 'diff':
        mag = np.mean(bspikes1, axis=1) - np.mean(bspikes2, axis=1)
    elif mag_meth == 'PI':
        mag = (
            (np.mean(bspikes1, axis=1) - np.mean(bspikes2, axis=1)) /
            (np.mean(bspikes1, axis=1) + np.mean(bspikes2, axis=1)))
    
    if fillval is None:
        fillval = np.nan
    
    is1d = False
    if bspikes1.ndim == 1:
        assert bspikes2.ndim == 1
        bspikes1 = bspikes1[None, :]
        bspikes2 = bspikes2[None, :]
        is1d = True
    
    if meth == 'ttest':        
        p = scipy.stats.ttest_ind(bspikes1, bspikes2, axis=1)[1]
    elif meth == 'sqrt_ttest':
        p = scipy.stats.ttest_ind(np.sqrt(bspikes1), np.sqrt(bspikes2), 
            axis=1)[1]
    elif meth == 'mannwhitneyu':
        p_l = []
        for row1, row2 in zip(bspikes1, bspikes2):
            if ~np.any(row1) and ~np.any(row2):
                p_l.append(fillval)
            elif np.all(np.sort(row1) == np.sort(row2)):
                p_l.append(fillval)
            else:
                p_l.append(2 * scipy.stats.mannwhitneyu(row1, row2)[1])
        p = np.array(p_l)
    else:
        raise ValueError("%s not accepted method" % meth)
    
    if p_adj_meth is not None:
        p = myutils.r_adj_pval(p, p_adj_meth)
    
    if is1d:
        return mag, p[0]
    else:
        return mag, p
    

def plot_effect_size_by_sound(fdf, fig=None, p_thresh=.05, **kwargs):
    """Wrapper around calc_effect_size_by_sound and a masked heatmap plot"""
    
    mag_d, p_d, names, t = calc_effect_size_by_sound(fdf, **kwargs)
    
    masked_heatmaps_by_sound(mag_d, p_d, t, fig=fig, p_thresh=p_thresh)
    
    return mag_d, p_d, names, t

def masked_heatmaps_by_sound(mag_d, p_d, t, fig=None, p_thresh=.05,
    clim=None, cmap=None):
    """Plot mag_d heatmap, masked by p_d, for each stimulus."""    
    cm = 0.0
    
    if fig is None:
        fig = plt.figure()

    for n, sound_name in enumerate(['lehi', 'rihi', 'lelo', 'rilo']):
        # get axis
        try:
            ax = fig.axes[n]
        except IndexError:
            ax = fig.add_subplot(2, 2, n + 1)
        
        to_plot = mag_d[sound_name].copy()
        to_plot[p_d[sound_name] > p_thresh] = np.nan
        to_plot[np.isnan(p_d[sound_name])] = np.nan
        
        try:
            cm = max([cm, np.abs(to_plot[~np.isnan(to_plot)].flatten()).max()])
        except ValueError:
            cm = 0.0
        
        myutils.my_imshow(to_plot, ax=ax, x=t, cmap=cmap)
        ax.set_title(sound_name)
    
    for ax in fig.axes:
        plt.axes(ax)
        if clim is None:
            plt.clim((-cm, cm))
        else:
            plt.clim(clim)        
    
    plt.show()



def calc_effect_size_by_sound(fdf, trial_lister=None, 
    comp_meth='ttest', p_adj_meth='BH', split_on=None, split_on_filter=None,
    t_start=-.25, t_stop=.5, bins=75, mag_meth='diff'):
    """Calculates a heat map of effect size for each sound, masked by p-value.
    
    Groups spikes by :split_on:, eg nid. Then compares across blocks,
    using compare_rasters(comp_meth, p_adj_meth).
    
    fdf : flat spike times
    trial_list : SpikeSorter.list_trials_by_type
    fig : figure to plot into
    comp_meth : how to compare the binned spike times across  blocks
    
    Returns
        mag_d : dict of effect magnitude by sound name
        p_d : dict of p-value by sound name
        names : list of values of :split_on:
        t : time bins
    """
    names, mag_d, p_d = [], {}, {}
    if split_on is None:
        g = {None: fdf}.items()
    else:
        g = fdf.groupby(split_on)
    
    # Iterate over groups
    n_incl = 0
    for key, df in g:        
        if split_on_filter is not None and key not in split_on_filter:
            continue
        else:
            n_incl += 1
        
        # get session name
        session_l = np.unique(np.asarray(df.session))
        if len(session_l) != 1:
            print("error: must be exactly one session!")
            1/0
        session = session_l[0]     
        names.append(key)

        # iterate over sound * block
        g1 = df.groupby(['sound', 'block'])    
        for n, sound_name in enumerate(['lehi', 'rihi', 'lelo', 'rilo']):
            # Keep a running list in each sound
            if sound_name not in mag_d:
                mag_d[sound_name] = []
                p_d[sound_name] = []
            
            # iterate over blocks and get folded spike times
            fsdict = {}
            for block_name in ['LB', 'PB']:
                # get spikes form this sound * block
                try:
                    x = df.ix[g1.groups[sound_name, block_name]]            
                except KeyError:
                    # no spikes of this sound * block!
                    x = []
                
                # fold by trial
                folded_spikes = fold(x, trial_lister(session, 
                    sound=sound_name, block=block_name))
                
                # convert to bins
                ns, t = myutils.times2bins(folded_spikes, return_t=True,
                    t_start=t_start, t_stop=t_stop, bins=bins, f_samp=30000.)
                
                fsdict[block_name] = ns.transpose()
            
            # Do the comparison
            mag, p = compare_rasters(fsdict['LB'], fsdict['PB'],
                meth=comp_meth, p_adj_meth=p_adj_meth, mag_meth=mag_meth)
            mag_d[sound_name].append(mag)
            p_d[sound_name].append(p)
    
    for key in mag_d:
        mag_d[key] = np.array(mag_d[key])
        p_d[key] = np.array(p_d[key])
    
    if split_on_filter is not None and len(split_on_filter) != n_incl:
        print("warning: %d in filter but only found %d" % (len(split_on_filter),
            n_incl))
    
    return mag_d, p_d, names, t

def count_spikes_in_slice(fsd, t_start, t_stop, split_on, split_on_filter, 
    trial_counter, pivot_by_sound=True, F_SAMP=30000., include_trials='hits'):
    """Wrapper around bin_spikes which just returns specified bin"""
    # Bin data
    bsd = bin_flat_spike_data2(fsd, split_on=split_on, t_start=t_start,
        t_stop=t_stop, n_bins=1, F_SAMP=F_SAMP,
        trial_counter=trial_counter, split_on_filter=split_on_filter)

    # Extract count
    bsd['resp'] = bsd.counts / (t_stop - t_start) / bsd.trials
        
    if pivot_by_sound:
        bsd = bsd.pivot_table(rows=['session', 'unit'], 
            cols=['sound', 'block'], values='resp')
    
    return bsd

def get_unit_filter(ratname=None):
    data_dir = '/media/TBLABDATA/20111208_frame_data/'
    
    if ratname is None:
        rlist = ['CR17B', 'CR13A', 'CR12B']
    else:
        rlist = [ratname]
    
    SUs = pandas.DataFrame()
    for r in rlist:        
        SUs = SUs.append(pandas.load(os.path.join(data_dir, '%s_SUs' % r)))

    unit_filter = SUs.to_dict()['not_poor_units']
    for key in unit_filter:
        unit_filter[key] = myutils.list_intersection(unit_filter[key],
            SUs.ix[key]['auditory_units'])
    
    unit_filter2 = []
    for session, session_SUs in unit_filter.items():
        for unit in session_SUs:
            unit_filter2.append((session, unit))
    
    return sorted(unit_filter2)


def fold(x, trial_list):
    if len(x) == 0:
        return [np.array([]) for n in range(len(trial_list))]
    
    # group those spikes by the trial from which they came
    g2 = x.groupby('trial')
    
    # grab spikes by trial
    folded_spikes = []
    for trial_number in trial_list:
        try:
            spike_idxs = g2.groups[trial_number]
        except KeyError:
            spike_idxs = np.array([])
        folded_spikes.append(
            np.asarray(x.ix[spike_idxs]['adj_time']))
    
    # error check
    n_empty_trials = sum([len(s) == 0 for s in folded_spikes])
    assert (n_empty_trials + len(np.unique(np.asarray(x.trial)))) == len(trial_list)    
    
    return folded_spikes

def plot_psths_by_sound(df, plot_difference=True, split_on=None,
    mark_significance=False, plot_errorbars=True, p_adj_meth=None, 
    plot_all=False):
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
        pdata = df.pivot_table(rows=['time'], cols=(['sound', 'block'] + split_on),
            values=['counts', 'trials'], aggfunc=np.sum)
    
    # Double-check we didn't lose data
    #if len(df) != pdata['counts'].shape[0] * pdata['counts'].shape[1]:
    if len(df) != np.sum(~np.isnan(pdata['counts'].values)):    
        print("warning: refolded data doesn't match shape")
        print("avoid this warning by passing a single trace, or specifying merge column")
    
    # Iterate over sounds (one axis per sound)
    for n, sound_name in enumerate(['lehi', 'rihi', 'lelo', 'rilo']):
        # extract data
        LB_counts = pdata['counts'][sound_name]['LB'].dropna(axis=1).values.astype(np.int)
        LB_trials = pdata['trials'][sound_name]['LB'].dropna(axis=1).values.astype(np.int)
        PB_counts = pdata['counts'][sound_name]['PB'].dropna(axis=1).values.astype(np.int)
        PB_trials = pdata['trials'][sound_name]['PB'].dropna(axis=1).values.astype(np.int)
        
        assert(LB_counts.shape == LB_trials.shape)
        assert(PB_counts.shape == PB_trials.shape)
        assert(PB_counts.shape == LB_trials.shape)
        
        # get time vector and check for consistency
        # technically could be in arbitrary order, in which case will error
        times = pdata['counts'][sound_name]['LB'].index.values.astype(np.float)
        assert (times == pdata['trials'][sound_name]['LB'].index.values).all()
        assert (times == pdata['counts'][sound_name]['PB'].index.values).all()
        assert (times == pdata['trials'][sound_name]['PB'].index.values).all()
        
        # remove columns with no trials
        bad_cols1 = np.array([np.all(t == 0) for t in LB_trials.transpose()])
        bad_cols2 = np.array([np.all(t == 0) for t in PB_trials.transpose()])
        bad_cols = bad_cols1 | bad_cols2
        print("dropping %d sessions with insufficient trials" % np.sum(bad_cols))
        LB_trials = LB_trials[:, ~bad_cols]
        LB_counts = LB_counts[:, ~bad_cols]
        PB_trials = PB_trials[:, ~bad_cols]
        PB_counts = PB_counts[:, ~bad_cols]        
        
        assert LB_counts.shape[1] != 0
        assert PB_counts.shape[1] != 0
        
        # Create axis for this plot and plot means with errorbars
        ax = f.add_subplot(2, 2, n + 1)
        if plot_all:
            ax.plot(times, LB_counts / LB_trials.astype(np.float),
                label='LB', color='b')
            ax.plot(times, PB_counts / PB_trials.astype(np.float),
                label='PB', color='r')
        else:
            myutils.plot_mean_trace(ax=ax, x=times, 
                data=LB_counts / LB_trials.astype(np.float), 
                label='LB', color='b', axis=1, errorbar=True)
            myutils.plot_mean_trace(ax=ax, x=times, 
                data=PB_counts / PB_trials.astype(np.float), 
                label='PB', color='r', axis=1, errorbar=True)

        assert(LB_counts.shape == LB_trials.shape)
        assert(PB_counts.shape == PB_trials.shape)
        assert(PB_counts.shape == LB_trials.shape)


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
        
        ax.set_title(myutils.longname[sound_name])
        plt.legend(loc='best')

    plt.show()    
    return f