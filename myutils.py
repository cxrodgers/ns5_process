import sys
import matplotlib.pyplot as plt
import numpy as np

def my_imshow(C, x=None, y=None, ax=None):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    
    if x is None:
        x = range(C.shape[1])
    if y is None:
        y = range(C.shape[0])
    extent = x[0], x[-1], y[0], y[-1]
    plt.imshow(np.flipud(C), interpolation='nearest', extent=extent)
    ax.axis('auto')
    plt.show()

def list_intersection(l1, l2):
    return list(set.intersection(set(l1), set(l2)))

def list_union(l1, l2):
    return list(set.union(set(l1), set(l2)))


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
    
def times2bins(times, f_samp=None, t_start=None, t_stop=None, bins=10,
    return_t=False):
    
    # dimensionality
    is2d = True
    try:
        len(times[0])
    except IndexError:
        is2d = False

    # optionally convert units    
    if f_samp is not None:
        if is2d:
            times = np.array([t / f_samp for t in times])
        else:
            times = np.asarray(times) / f_samp
    
    # defaults for time
    if is2d:
        if t_start is None:
            t_start = min([x.min() for x in times])
        if t_stop is None:
            t_stop = max([x.max() for x in times])
    else:
        if t_start is None:
            t_start = times.min()
        if t_stop is None:
            t_stop = times.max()

    # determine spacing of time bins
    t_vals = np.linspace(t_start, t_stop, bins + 1)

    # histogram
    if not is2d:
        res = np.histogram(times, bins=t_vals)[0]
    else:
        res = np.array([np.histogram(x, bins=t_vals)[0] for x in times])

    if return_t:
        return res, t_vals
    else:
        return res

def plot_rasters(obj, ax=None, full_range=1.0, y_offset=0.0, plot_kwargs=None):
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
            y_offset + np.ones(trial_spikes.shape, dtype=np.float) * 
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