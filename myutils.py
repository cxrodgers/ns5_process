import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import mlab
import matplotlib.pylab
import scipy.stats
import matplotlib
import wave
import struct


longname = {'lelo': 'LEFT+LOW', 'rilo': 'RIGHT+LOW', 'lehi': 'LEFT+HIGH',
    'rihi': 'RIGHT+HIGH'}

def load_waveform_from_wave_file(filename, dtype=np.float, rescale=False,
    also_return_fs=False, never_flatten=False, mean_channel=False):
    """Opens wave file and reads, assuming signed shorts.
    
    if rescale, returns floats between -1 and +1
    if also_return_fs, returns (sig, f_samp); else returns sig
    if never_flatten, then returns each channel its own row
    if not never_flatten and only 1 channel, returns 1d array
    if mean_channel, return channel mean (always 1d)
    """
    wr = wave.Wave_read(filename)
    nch = wr.getnchannels()
    nfr = wr.getnframes()
    sig = np.array(struct.unpack('%dh' % (nfr*nch), wr.readframes(nfr)), 
        dtype=dtype)
    wr.close()
    
    # reshape into channels
    sig = sig.reshape((nfr, nch)).transpose()
    
    if mean_channel:
        sig = sig.mean(axis=0)
    
    if not never_flatten and nch == 1:
        sig = sig.flatten()
    
    if rescale:
        sig = sig / 2**15
    
    if also_return_fs:
        return sig, wr.getframerate()
    else:
        return sig   

class UniqueError(Exception):
    pass

def unique_or_error(a):
    """Asserts that `a` contains only one unique value and returns it"""    
    u = np.unique(np.asarray(a))
    if len(u) == 0:
        raise UniqueError("no values found")
    if len(u) > 1:
        raise UniqueError("%d values found, should be one" % len(u))
    else:
        return u[0]

def only_one(l):
    """this will be redefined to error unless `l` is length one"""
    print "warning: use `unique_or_error`"
    ll = np.unique(np.asarray(l))
    assert len(ll) == 1, "values are not unique"
    return ll[0]

def plot_with_trend_line(x, y, xname='X', yname='Y', ax=None):
    dropna = np.isnan(x) | np.isnan(y)
    x = x[~dropna]
    y = y[~dropna]
    
    if ax is None:    
        f = plt.figure()
        ax = f.add_subplot(111)
    ax.plot(x, y, '.')
    #p = scipy.polyfit(x.flatten(), y.flatten(), deg=1)    
    m, b, rval, pval, stderr = \
        scipy.stats.stats.linregress(x.flatten(), y.flatten())
    ax.plot([x.min(), x.max()], m * np.array([x.min(), x.max()]) + b, 'k:')
    plt.legend(['Trend r=%0.3f p=%0.3f' % (rval, pval)], loc='best')
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    plt.show()


def polar_plot_by_sound(Y, take_sqrt=False, normalize=False, ax=None, **kwargs):
    """Y should have 4 columns in it, one for each sound."""
    if hasattr(Y, 'index'):
        YY = Y[['rihi', 'lehi', 'lelo', 'rilo']].values.transpose()
    else:
        YY = Y.transpose()
    
    if YY.ndim == 1:
        YY = YY[:, np.newaxis]

    if normalize:
        YY = np.transpose([row / row.mean() for row in YY.transpose()])

    YY[YY < 0.0] = 0.0
    if take_sqrt:
        YY = np.sqrt(YY)
    
    

    YYY = np.concatenate([YY, YY[0:1, :]])

    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111, polar=True)

    ax.plot(np.array([45, 135, 225, 315, 405])*np.pi/180.0, YYY, **kwargs)


def prefidx(A, B):
    return (A - B) / (A + B)

class Spectrogrammer:
    """Turns a waveform into a spectrogram"""
    def __init__(self, NFFT=256, downsample_ratio=5, new_bin_width_sec=None,
        max_freq=40e3, min_freq=5e3, Fs=200e3, noverlap=None, normalization=1.0,
        detrend=matplotlib.pylab.detrend_mean):
        """Initialize object to turn waveforms to spectrograms.
        
        Stores parameter choices, so you can batch analyze waveforms using
        the `transform` method.
        
        If you specify new_bin_width_sec, this chooses the closest integer 
        downsample_ratio and that parameter is actually saved and used.
        
        TODO: catch other kwargs and pass to specgram.
        """
        
        # figure out downsample_ratio
        if new_bin_width_sec is not None:
            self.downsample_ratio = int(np.rint(new_bin_width_sec * Fs / NFFT))
        else:
            self.downsample_ratio = int(downsample_ratio)
        
        if self.downsample_ratio == 0:
            print "requested temporal resolution too high, using maximum"
            self.downsample_ratio = 1
            
        
        # store other defaults
        self.NFFT = NFFT
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.Fs = Fs
        self.noverlap = noverlap
        if self.noverlap is None:
            self.noverlap = NFFT / 2
        
        self.normalization = normalization
        self.detrend = detrend

    
    def transform(self, waveform):
        """Converts a waveform to a suitable spectrogram.
        
        Removes high and low frequencies, rebins in time (via median)
        to reduce data size. Returned times are the midpoints of the new bins.
        
        Returns:  Pxx, freqs, t    
        Pxx is an array of dB power of the shape (len(freqs), len(t)).
        It will be real but may contain -infs due to log10
        """
        # For now use NFFT of 256 to get appropriately wide freq bands, then
        # downsample in time
        Pxx, freqs, t = mlab.specgram(waveform, NFFT=self.NFFT, 
            noverlap=self.noverlap, Fs=self.Fs, detrend=self.detrend)
        Pxx = Pxx * np.tile(freqs[:, np.newaxis] ** self.normalization, 
            (1, Pxx.shape[1]))

        # strip out unused frequencies
        Pxx = Pxx[(freqs < self.max_freq) & (freqs > self.min_freq), :]
        freqs = freqs[(freqs < self.max_freq) & (freqs > self.min_freq)]

        # Rebin in size "downsample_ratio". If last bin is not full, discard.
        Pxx_rebinned = []
        t_rebinned = []
        for n in range(0, len(t) - self.downsample_ratio + 1, 
            self.downsample_ratio):
            Pxx_rebinned.append(
                np.median(Pxx[:, n:n+self.downsample_ratio], axis=1).flatten())
            t_rebinned.append(
                np.mean(t[n:n+self.downsample_ratio]))

        # Convert to arrays
        Pxx_rebinned_a = np.transpose(np.array(Pxx_rebinned))
        t_rebinned_a = np.array(t_rebinned)

        # log it and deal with infs
        Pxx_rebinned_a_log = -np.inf * np.ones_like(Pxx_rebinned_a)
        Pxx_rebinned_a_log[np.nonzero(Pxx_rebinned_a)] = \
            10 * np.log10(Pxx_rebinned_a[np.nonzero(Pxx_rebinned_a)])


        self.freqs = freqs
        self.t = t_rebinned_a
        return Pxx_rebinned_a_log, freqs, t_rebinned_a

def set_fonts_big(undo=False):
    if not undo:
        matplotlib.rcParams['font.size'] = 16.0
        matplotlib.rcParams['xtick.labelsize'] = 'medium'
        matplotlib.rcParams['ytick.labelsize'] = 'medium'
    else:
        matplotlib.rcParams['font.size'] = 10.0
        matplotlib.rcParams['xtick.labelsize'] = 'small'
        matplotlib.rcParams['ytick.labelsize'] = 'small'

def my_imshow(C, x=None, y=None, ax=None, cmap=None):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    
    if x is None:
        x = range(C.shape[1])
    if y is None:
        y = range(C.shape[0])
    extent = x[0], x[-1], y[0], y[-1]
    plt.imshow(np.flipud(C), interpolation='nearest', extent=extent, cmap=cmap)
    ax.axis('auto')
    plt.show()

def iziprows(df):
   series = [df[col] for col in df.columns]
   series.insert(0, df.index)
   return itertools.izip(*series)

def list_intersection(l1, l2):
    return list(set.intersection(set(l1), set(l2)))

def list_union(l1, l2):
    return list(set.union(set(l1), set(l2)))


def parse_space_sep(s, dtype=np.int):
    """Returns a list of integers from a space-separated string"""
    s2 = s.strip()
    if s2 == '':
        return []    
    else:
        return [dtype(ss) for ss in s2.split()]

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
    
    return np.std(np.asarray(data), axis) / np.sqrt(N)

def printnow(s):
    """Write string to stdout and flush immediately"""
    sys.stdout.write(str(s) + "\n")
    sys.stdout.flush()

def plot_mean_trace(ax=None, data=None, x=None, errorbar=True, axis=0, **kwargs):
    
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
    
    data = np.asarray(data)
    
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
            x = range(len(np.mean(data, axis=axis)))
    
    if single_trace:
        ax.plot(x, data, **kwargs)
    else:
        if errorbar:
            ax.errorbar(x=x, y=np.mean(data, axis=axis),
                yerr=std_error(data, axis=axis), **kwargs)
        else:
            ax.plot(np.mean(data, axis=axis), **kwargs)

def plot_asterisks(pvals, ax, x=None, y=0, yd=1, levels=None):
    pvals = np.asarray(pvals)
    if levels is None:
        levels = [.05, .01, .001, .0001]
    if x is None:
        x = range(len(pvals))
    x = np.asarray(x)

    already_marked = np.zeros(len(pvals), dtype=np.bool)
    for n, l in enumerate(levels):
        msk = (pvals < l)# & ~already_marked
        if np.any(msk):
            ax.plot(x[np.where(msk)[0]], n*yd + y * np.ones_like(np.where(msk)[0]),
                marker='*', color='k', ls='None')
        #already_marked = already_marked | msk
    plt.show()

def times2bins(times, f_samp=None, t_start=None, t_stop=None, bins=10,
    return_t=False):
    
    # dimensionality
    is2d = True
    try:
        len(times[0])
    except (TypeError, IndexError):
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

def histogram_pvals(eff, p, bins=20, thresh=.05, ax=None):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    
    if np.sum(p > thresh) == 0:
        ax.hist(eff[p<=thresh], bins=bins, histtype='barstacked', color='r')    
    elif np.sum(p < thresh) == 0:
        ax.hist(eff[p>thresh], bins=bins, histtype='barstacked', color='k')            
    else:
        ax.hist([eff[p>thresh], eff[p<=thresh]], bins=bins, 
            histtype='barstacked', color=['k', 'r'], rwidth=1.0)    
    plt.show()

def sort_df_by_col(a, col):
    return a.ix[a.index[np.argsort(np.asarray(a[col]))]]

def pick_mask(df, **kwargs):
    """Returns mask of df, s.t df[mask][key] == val for key, val in kwargs
    """
    mask = np.ones(len(df), dtype=np.bool)
    for key, val in kwargs.items():
        mask = mask & (df[key] == val)
    
    return mask

def pick_count(df, **kwargs):
    return np.sum(pick_mask(df, **kwargs))


def polar_plot_by_sound(Y, take_sqrt=False, normalize=False, ax=None, **kwargs):
    """Y should have 4 columns in it, one for each sound."""
    if hasattr(Y, 'index'):
        YY = Y[['rihi', 'lehi', 'lelo', 'rilo']].values.transpose()
    else:
        YY = Y.transpose()
    
    if YY.ndim == 1:
        YY = YY[:, np.newaxis]

    if normalize:
        YY = np.transpose([row / row.mean() for row in YY.transpose()])

    YY[YY < 0.0] = 0.0
    if take_sqrt:
        YY = np.sqrt(YY)
    
    

    YYY = np.concatenate([YY, YY[0:1, :]])

    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111, polar=True)

    ax.plot(np.array([45, 135, 225, 315, 405])*np.pi/180.0, YYY, **kwargs)
    
    
    plt.xticks(np.array([45, 135, 225, 315, 405])*np.pi/180.0,
        ['RIGHT+HIGH', 'LEFT+HIGH', 'LEFT+LOW', 'RIGHT+LOW'])