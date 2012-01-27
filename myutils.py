import sys
import matplotlib.pyplot as plt
import numpy as np


def parse_space_sep(s):
    """Returns a list of integers from a space-separated string"""
    s2 = s.strip()
    if s2 == '':
        return []    
    else:
        return [int(ss) for ss in s2.split()]

def r_adj_pval(a, meth='BH'):
    import rpy2.robjects as robjects
    r = robjects.r
    robjects.globalenv['unadj_p'] = robjects.FloatVector(
        np.asarray(a).flatten())
    return np.array(r("p.adjust(unadj_p, '%s')" % meth)).reshape(a.shape)


def std_error(data, axis=None):
    if axis is None:
        N = len(data)
    else:
        N = np.asarray(data).shape[axis]
    
    return np.std(data, axis) / np.sqrt(N)

def printnow(s):
    """Write string to stdout and flush immediately"""
    sys.stdout.write(s + "\n")
    sys.stdout.flush()

def plot_mean_trace(ax=None, data=None, x=None, errorbar=True, axis=0, **kwargs):
    
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
    
    if np.min(data.shape) == 1:
        data = data.flatten()
    if data.ndim == 1:
        single_trace = True
        errorbar = False
        
        if x is None:
            x = range(len(data))
    else:
        single_trace = False
        
        if x is None:
            x = range(data.shape[axis])
    
    if single_trace:
        ax.plot(x, data, **kwargs)
    else:
        if errorbar:
            ax.errorbar(x=x, y=np.mean(data, axis=axis),
                yerr=std_error(data, axis=axis), **kwargs)
        else:
            ax.plot(np.mean(data, axis=axis), **kwargs)
    
    


def plot_rasters(obj, ax=None, full_range=1.0, plot_kwargs=None):
    """Plots raster of spike times or psth object.
    
    obj : PSTH object, or array of spike times (in seconds)
    ax : axis object to plot into
    plot_kwargs : any additional plot specs. Defaults:
        if 'color' not in plot_kwargs: plot_kwargs['color'] = 'k'
        if 'ms' not in plot_kwargs: plot_kwargs['ms'] = 4
        if 'marker' not in plot_kwargs: plot_kwargs['marker'] = '|'
        if 'ls' not in plot_kwargs: plot_kwargs['ls'] = 'None'    
    full_range: y-value of top row (last trial), default 1.0
    
    Assumes that spike times are aligned to each trial start, and uses
    this to guess where trial boundaries are: any decrement in spike
    time is assumed to be a trial boundary. This will sometimes lump
    trials together if there are very few spikes.
    """
    # get spike times
    try:
        ast = obj.adjusted_spike_times / float(obj.F_SAMP)
    except AttributeError:
        ast = obj    
    
    try:
        # is it already folded?
        len(ast[0])
        folded_spike_times = ast
    except TypeError:
        # need to fold
        # convert into list representation
        folded_spike_times = fold_spike_times(ast)

    # build axis
    if ax is None:
        f = plt.figure(); ax = f.add_subplot(111)
    
    # plotting defaults
    if plot_kwargs is None:
        plot_kwargs = {}
    if 'color' not in plot_kwargs: plot_kwargs['color'] = 'k'
    if 'ms' not in plot_kwargs: plot_kwargs['ms'] = 4
    if 'marker' not in plot_kwargs: plot_kwargs['marker'] = '|'
    if 'ls' not in plot_kwargs: plot_kwargs['ls'] = 'None'
    
    if full_range is None:
        full_range = float(len(folded_spike_times))
    
    for n, trial_spikes in enumerate(folded_spike_times):
        ax.plot(trial_spikes, 
            np.ones(trial_spikes.shape, dtype=np.float) * 
            n / float(len(folded_spike_times)) * full_range,
            **plot_kwargs)


def pick_mask(df, **kwargs):
    """Returns mask of df, s.t df[mask][key] == val for key, val in kwargs
    """
    mask = np.ones(len(df), dtype=np.bool)
    for key, val in kwargs.items():
        mask = mask & (df[key] == val)
    
    return mask

def pick_count(df, **kwargs):
    return np.sum(pick_mask(df, **kwargs))