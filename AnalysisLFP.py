from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
import os.path
import glob
import pandas
from . import SpikeAnalysis
from . import myutils
from matplotlib import mlab
import scipy.stats


class MultiServeLFP(object):
    def __init__(self, lfp_dir_list=None, tdf_filename_list=None, filter_list=None):
        self.lfp_dir_list = lfp_dir_list
        self.tdf_filename_list = tdf_filename_list
        self.ssl_list = None
        self.filter_list = filter_list
    
    def refresh_files(self):
        self.ssl_list = []
        
        if self.filter_list is not None:
            fla = np.array(self.filter_list)

        for lfp_dir, tdf_filename in zip(self.lfp_dir_list, 
            self.tdf_filename_list):
            ssl = SingleServeLFP(dirname=lfp_dir, tdf_file=tdf_filename)
            
            if self.filter_list is not None:
                keep_tets = fla[fla[:, 0] == ssl.session][:, 1].astype(np.int)
            else:
                keep_tets = None
            ssl.split_on_filter = keep_tets
            
            self.ssl_list.append(ssl)

        self.session_list = [ssl.session for ssl in self.ssl_list]
    
    def read(self):
        self.refresh_files()
        
        
        ddf = pandas.DataFrame()
        bins = None
        tt = None
        for ssl in self.ssl_list:
            df, t = ssl.read(return_t=True, include_trials='hits')
            if len(df) > 0:
                bins = ssl.bins
                tt = ssl.t
            ddf = ddf.append(df, ignore_index=True)
        self._lfp = ddf
        self.t = tt
        self.bins = bins
    
    def average_grouped_by_sound(self, detrend=None, **kwargs):
        # Optionally filter by keywork, eg tetrode=[2]
        if len(kwargs) > 0:
            lfpf = self.lfp.copy()
            for key, val in list(kwargs.items()):
                lfpf = lfpf[lfpf[key].isin(val)]
        else:
            lfpf = self.lfp
        
        # Iterate over sound * block
        g = lfpf.groupby(['sound', 'block'])
        res = {}
        for sound in ['lehi', 'rihi', 'lelo', 'rilo']:
            res[sound] = {}
            for block in ['LB', 'PB']:
                df = lfpf.ix[g.groups[sound, block]]
                
                # Now try to group by session * tetrode
                g2 = df.groupby(['session', 'tetrode'])
                if len(g2.groups) == 1:
                    # Only a single session * tetrode, plot across trials                
                    res[sound][block] = df[self.bins]
                else:
                    ddf = pandas.DataFrame()
                    res[sound][block] = np.array([
                        df.ix[val][self.bins].mean(axis=0)
                        for val in list(g2.groups.values())])
                
                # Optional detrend
                if detrend == 'baseline':
                    pre_bins = self.bins[self.t < 0.0]
                    baseline = res[sound][block][pre_bins].mean(axis=1)
                    tb = np.tile(baseline[:, np.newaxis], (1, len(self.t)))
                    res[sound][block] -= tb
                elif detrend is not None:
                    raise ValueError("detrend mode not supported")
        return res
    
    
    @property
    def lfp(self):
        if hasattr(self, '_lfp'):
            return self._lfp
        else:
            self.read()
            return self._lfp
    

class SingleServeLFP(object):
    def __init__(self, dirname=None, filename_filter=None, tdf_file=None,
        split_on_filter=None):
        self.dirname = dirname
        if dirname is not None:
            self.session = os.path.split(self.dirname)[1]
        self.filename_filter = filename_filter
        self.tdf_file = tdf_file
        self.split_on_filter = split_on_filter
        
        if self.split_on_filter is None:
            self.split_on_filter = list(range(16))

    def refresh_files(self):
        self.lfp_filenames = []
        self.lfp_tetrodes = []
        if self.split_on_filter is None:
            self.split_on_filter = list(range(16))
        
        
        if self.filename_filter is None:
            putative_files = sorted(glob.glob(os.path.join(
                self.dirname, '*')))
        else:
            putative_files = sorted(glob.glob(os.path.join(
                self.dirname, ('*%s*' % self.filename_filter))))
        
        for fn in putative_files:
            for tetrode in self.split_on_filter:
                m = glob.re.search('\.lfp\.%d\.npz$' % tetrode, fn)
                if m is not None:
                    self.lfp_filenames.append(fn)
                    self.lfp_tetrodes.append(tetrode)
        
        self._load_tdf()
    
    def _load_tdf(self):
        try:
            self._tdf = pandas.load(self.tdf_file)
        except IOError:
            print("warning: cannot load trials %s" % self.tdf_file)
            self._tdf = None
    
    def _read_numpy_z_format(self, fn):
        nz = np.load(fn)        
        lfp = nz['lfp']
        stored_trial_numbers = nz['trial_numbers']        
        N = nz['lfp'].shape[0]
        if len(stored_trial_numbers) > N:
            # sometimes stored duplicate!
            assert np.all(stored_trial_numbers[:N] == stored_trial_numbers[N:])
            stored_trial_numbers = stored_trial_numbers[:N]        
        t = nz['t']        
        nz.close()
        return lfp, t, stored_trial_numbers

    def read(self, return_t=False, include_trials='hits', stim_number_filter=None):
        """Return DataFrame containing all LFP from this session"""
        if stim_number_filter is None:
            stim_number_filter = list(range(5, 13))
        
        # Search directory for files
        self.refresh_files()
        
        bigdf = pandas.DataFrame()
        t_vals = None
        bins = None
        for lfp_filename, tetrode in zip(self.lfp_filenames, self.lfp_tetrodes):
            # Load numpy z format
            lfp, t, trial_numbers = self._read_numpy_z_format(lfp_filename)        
            if t_vals is None:
                t_vals = t
            else:
                assert np.all(t_vals - t < 1e-7)                
            bins = np.array(['t%d' % n for n in range(lfp.shape[1])])
            df = pandas.DataFrame(lfp, columns=bins)
            df.insert(loc=0, column='trial', value=trial_numbers)
            df.insert(loc=0, column='tetrode', value=tetrode)
            df.insert(loc=0, column='session', 
                value=[os.path.split(self.dirname)[1]]*len(df))
            bigdf = bigdf.append(df, ignore_index=True)
        
        self.t = t_vals
        self.bins = bins
        
        if len(bigdf) > 0 and self.tdf is not None:
            bigdf = bigdf.join(self.tdf[
                ['block', 'outcome', 'stim_number', 't_center', 'nonrandom']], on='trial')
            
            if include_trials == 'hits':
                bigdf = bigdf[(bigdf.outcome == 1) & (bigdf.nonrandom == 0)]            
                bigdf.pop('nonrandom')
                bigdf.pop('outcome')
            
            bigdf = bigdf[bigdf.stim_number.isin(stim_number_filter)]
            
            SpikeAnalysis.replace_stim_numbers_with_names(bigdf)
        
        if return_t:
            return bigdf, self.t
        else:
            return bigdf
    
    @property
    def lfp(self):
        if hasattr(self, '_lfp'):
            return self._lfp
        else:
            self._lfp = self.read()
            return self._lfp
    
    @property
    def tdf(self):
        if hasattr(self, '_tdf'):
            return self._tdf
        else:
            self._load_tdf()
            return self._tdf

    def average_grouped_by_sound(self, detrend=None, **kwargs):
        # Optionally filter by keywork, eg tetrode=[2]
        if len(kwargs) > 0:
            lfpf = self.lfp.copy()
            for key, val in list(kwargs.items()):
                lfpf = lfpf[lfpf[key].isin(val)]
        else:
            lfpf = self.lfp
        
        # Iterate over sound * block
        g = lfpf.groupby(['sound', 'block'])
        res = {}
        for sound in ['lehi', 'rihi', 'lelo', 'rilo']:
            res[sound] = {}
            for block in ['LB', 'PB']:
                df = lfpf.ix[g.groups[sound, block]]
                
                # Now try to group by session * tetrode
                g2 = df.groupby(['session', 'tetrode'])
                if len(g2.groups) == 1:
                    # Only a single session * tetrode, plot across trials                
                    res[sound][block] = df[self.bins]
                else:
                    ddf = pandas.DataFrame()
                    res[sound][block] = np.array([
                        df.ix[val][self.bins].mean(axis=0)
                        for val in list(g2.groups.values())])
                
                # Optional detrend
                if detrend == 'baseline':
                    pre_bins = self.bins[self.t < 0.0]
                    baseline = res[sound][block][pre_bins].mean(axis=1)
                    tb = np.tile(baseline[:, np.newaxis], (1, len(self.t)))
                    res[sound][block] -= tb
                elif detrend is not None:
                    raise ValueError("detrend mode not supported")
        return res
    

def plot_lfp_grouped_by_sound(ssl, plot_difference=True, p_adj_meth=None,
    mark_significance=True, t_start=None, t_stop=None, **kwargs):
    # First get grouped averages
    res_d = ssl.average_grouped_by_sound(**kwargs)
    
    # set time limits
    if t_start is None:
        t_start = ssl.t[0]
    if t_stop is None:
        t_stop = ssl.t[-1]
    t1bin = np.argmin(np.abs(ssl.t - t_start))
    t2bin = np.argmin(np.abs(ssl.t - t_stop)) + 1
    
    f = plt.figure()
    for n, sound in enumerate(['lehi', 'rihi', 'lelo', 'rilo']):
        # Create an axis for this sound and plot both blocks
        ax = f.add_subplot(2, 2, n+1)
        for block in ['LB', 'PB']:
            myutils.plot_mean_trace(ax=ax, x=ssl.t[t1bin:t2bin], 
                data=res_d[sound][block][:, t1bin:t2bin], label=block)
        
        # Optionally plot difference
        if plot_difference:            
            di = res_d[sound]['LB'] - res_d[sound]['PB']
            myutils.plot_mean_trace(ax=ax, x=ssl.t[t1bin:t2bin], 
                data=di[:, t1bin:t2bin], label='diff', color='m')
        
        # Optionally mark significance
        if mark_significance:
            p_vals = scipy.stats.ttest_rel(res_d[sound]['LB'][:, t1bin:t2bin],
                res_d[sound]['PB'][:, t1bin:t2bin])[1]            
            if p_adj_meth is not None:
                p_vals = myutils.r_adj_pval(p_vals, meth=p_adj_meth)            
            pp = np.where(p_vals < .05)[0]
            plt.plot(ssl.t[t1bin:t2bin][pp], np.zeros_like(pp), 'k*')            
            pp = np.where(p_vals < .01)[0]
            plt.plot(ssl.t[t1bin:t2bin][pp], np.zeros_like(pp), 'ko',
                markerfacecolor='w')                    
        
        plt.legend(loc='best')
        ax.set_title(sound)
        ax.set_xlim((t_start, t_stop))
    plt.show()

def get_tetrode_filter(ratname=None):
    fn_d = {
        'CR12B': '/media/STELLATE/20111208_CR12B_allsessions_sorted/data_params_CR12B.csv',
        'CR17B': '/media/STELLATE/20110907_CR17B_allsessions_sorted/data_params_CR17B.csv',
        'CR13A': '/media/STELLATE/20110816_CR13A_allsessions_sorted/data_params_CR13A.csv'
        }    
    
    if ratname is None:
        l = []
        for r in list(fn_d.keys()):
            l += get_tetrode_filter(r)        
        return l

    dp = mlab.csv2rec(fn_d[ratname])
    tetrode_filter = []
    for row in dp:
        if row['session_type'] != 'behaving':
            continue
        for t in myutils.parse_space_sep(row['auditory_tetrodes']):
            tetrode_filter.append((row['session_name'], t))    
    return sorted(tetrode_filter)

def get_subdir_list(ratname):
    fn_d = {
        'CR12B': '/media/STELLATE/20111208_CR12B_allsessions_sorted/*behaving',
        'CR17B': '/media/STELLATE/20110907_CR17B_allsessions_sorted/*behaving',
        'CR13A': '/media/STELLATE/20110816_CR13A_allsessions_sorted/*behaving',
        }    
    return sorted(glob.glob(fn_d[ratname]))

def build_tdf_filename_list(subdir_list):
    session_list = [os.path.split(subdir)[1] for subdir in subdir_list]
    tdf_file_list = ['/media/TBLABDATA/20111208_frame_data/%s_trials' % session
        for session in session_list]
    return tdf_file_list