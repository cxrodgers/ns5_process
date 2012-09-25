import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import mlab
import matplotlib.pylab
import scipy.stats
import matplotlib
import wave
import struct
import os.path
import datetime
import scipy.io
import scipy.signal
from lxml import etree
import kkpandas

longname = {'lelo': 'LEFT+LOW', 'rilo': 'RIGHT+LOW', 'lehi': 'LEFT+HIGH',
    'rihi': 'RIGHT+HIGH'}
LBPB_short2long = longname
LBPB_sns = list(range(1, 13))
LBPB_stimnames = [
    'lo_pc_go', 'hi_pc_no', 'le_lc_go', 'ri_lc_no',
    'le_hi_pc', 'ri_hi_pc', 'le_lo_pc', 'ri_lo_pc',
    'le_hi_lc', 'ri_hi_lc', 'le_lo_lc', 'ri_lo_lc']
LBPB_sn2name = {k: v for k, v in zip(LBPB_sns, LBPB_stimnames)}


def pickle_dump(obj, filename):
    import cPickle
    with file(filename, 'w') as fi:
        cPickle.dump(obj, fi)

def pickle_load(filename):
    import cPickle
    with file(filename) as fi:
        res = cPickle.load(fi)
    return res

def invert_linear_poly(p):
    return np.array([1, -p[1]]).astype(np.float) / p[0]

def soundsc(waveform, fs=44100, normalize=True):
    import scikits.audiolab as skab
    waveform = np.asarray(waveform)
    n = waveform.astype(np.float) / np.abs(waveform).max()
    skab.play(n, fs)


class GaussianSmoother:
    def __init__(self, filter_std=5, width=10, gain=1):
        self.filter_std = float(filter_std)
        self.width = float(width)
        self.gain = float(gain)
        
        self.n = np.arange(filter_std * width, dtype=np.int)
        self.b = np.exp( -(self.n.astype(np.float) ** 2) / (2 * filter_std**2) )
        self.b = self.gain * self.b / np.sqrt((self.b ** 2).sum())
        self.a = np.array([1])
    
    def execute(self, input_data, **filter_kwargs):
        input_data = np.asarray(input_data)
        res = scipy.signal.filtfilt(self.b, self.a, input_data, **filter_kwargs)
        return res
    



class ToneLoader:
    
    def __init__(self, filename=None):
        self.filename = filename
        self.data_dict = None
        
        if self.filename is not None:
            self._load()
    
    def _load(self):
        self._parse_date()
        self.data_dict = scipy.io.loadmat(self.filename)
    
    def _parse_date(self):
        """Split filename on _ and set date field"""
        split_fields = os.path.split(self.filename)[1].split('_')[1:]
        split_fields[-1] = os.path.splitext(split_fields[-1])[0]
        
        # convert strings to integers, will crash here if parsing wrong
        self.date_fields = map(int, split_fields)
        
        self.datetime = datetime.datetime(*self.date_fields)
    
    def aliased_tones(self, Fs=30e3, take_abs=True):
        """Returns the tone frequencies as they would appear aliased"""
        start = self.tones        
        aliased = np.mod(start + Fs/2., Fs) - Fs/2.
        
        if take_abs:
            return np.abs(aliased)
        else:
            return aliased
    
    @property
    def tones(self):
        if self.data_dict is None:
            self._load()
        return self.data_dict['tones'].flatten()
    
    @property
    def attens(self):
        if self.data_dict is None:
            self._load()
        return self.data_dict['attens'].flatten()
    
    def __repr__(self):
        return "ToneLoader('%s')" % self.filename


def parse_bitstream(bitstream, onethresh=.7 * 2**15, zerothresh=.3 * 2**15,
    min_interword=190, bitlen=10, certainty_thresh=7, nbits=16, 
    debug_mode=False):
    """Parse digital words from an analog trace and return times + values.
    
    This is for asynchronous digital communication, meaning a digital word
    is sent at unknown times. It is assumed that each word begins with a high
    bit ("1") to indicate when parsing should begin. Thereafter bits are read
    off in chunks of length `bitlen` and decoded to 1 or 0.
    
    Finally the `nbits` sequential bits are converted to an integer value
    by assuming LSB last, and subtracting the start bit. Example:
        1000000000000100 => 4
    
    Arguments
    ---------
    bitstream : analog trace
    onethresh : minimum value to decode a one
    zerothresh : maximum value to decode a zero
    min_interword : reject threshold crossings that occur more closely spaced 
        than this. Default is slightly longer than anticipated word duration 
        to avoid edge cases. Minimal error checking is done so this will not
        work for noisy signals -- spurious voltage transients could be decoded
        as zeros and potentially mask subsequent words within `min_interword`.
    bitlen : duration of each decoded bit, in samples
    certainty_thresh : number of samples per decoded bit necessary to decode
        it. That is, at least this many samples out of `bitlen` samples need
        to be above onethresh XOR below zerothresh. An error occurs if this
        threshold is not met.
    nbits : number of decoded bits per word
    debug_mode : plot traces of each word
    
    Returns times, numbers:
        times : times in samples of ONSET of each word
        numbers : value of each word
    """
    # Dot product the decoded bits with this to convert to integer
    wordconv = 2**np.arange(nbits, dtype=np.int)[::-1]

    # Threshold the signal
    ones = np.where(bitstream > onethresh)[0]
    #zeros = np.where(bitstream < zerothresh)[0] # never actually used?

    # Return empties if no ones found
    if len(ones) == 0:
        return np.array([]), np.array([])

    # Find when start bits occur, rejecting those within refractory period
    trigger_l = [ones[0]]
    for idx in ones[1:]:
        if idx > trigger_l[-1] + min_interword:
            trigger_l.append(idx)
    trial_start_times = np.asarray(trigger_l, dtype=np.int)

    # Plot if necessary
    if debug_mode:
        plt.figure()
        for trial_start in trial_start_times:
            plt.plot(bitstream[trial_start:trial_start+min_interword])
        plt.show()

    # Decode bits from each word
    trial_numbers = []
    for trial_start in trial_start_times:
        word = []
        # Extract one bit at a time
        for nbit in range(nbits):
            bitstr = bitstream[trial_start + nbit*bitlen + range(bitlen)]
            
            # Decode as 1, 0, or unknown
            if np.sum(bitstr > onethresh) > certainty_thresh:
                bitchoice = 1
            elif np.sum(bitstr < zerothresh) > certainty_thresh:
                bitchoice = 0
            else:
                bitchoice = -1
            word.append(bitchoice)
        
        # Fail if unknown bits occurred
        if -1 in word:
            1/0
        
        # Convert to integer
        val = np.sum(wordconv * np.array(word))
        trial_numbers.append(val)
    trial_numbers = np.asarray(trial_numbers, dtype=np.int)
    
    # Drop the high bit signal
    trial_numbers = trial_numbers - (2**(nbits-1))
    
    return trial_start_times, trial_numbers


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

def wav_write(filename, signal, dtype=np.int16, rescale=True, fs=8000):
    """Write wave file to filename.
    
    If rescale:
        Assume signal is between -1.0 and +1.0, and will multiply by maximum
        of the requested datatype. Otherwise `signal` will be converted
        to dtype as-is.
    
    If signal.ndim == 1:
        writes mono
    Elif signal.ndim == 2:
        better be stereo with each row a channel
    """
    if signal.ndim == 1:
        nchannels = 1
    elif signal.ndim == 2:
        nchannels = 2
        signal = signal.transpose().flatten()
    
    assert signal.ndim == 1, "only mono supported for now"
    assert dtype == np.int16, "only 16-bit supported for now"
    
    # rescale and convert signal
    if rescale:
        factor = np.iinfo(dtype).max
        sig = np.rint(signal * factor).astype(dtype)
    else:
        sig = sig.astype(dtype)
    
    # pack (don't know how to make this work for general dtype
    sig = struct.pack('%dh' % len(sig), *list(sig))
    
    # write to file
    ww = wave.Wave_write(filename)
    ww.setnchannels(nchannels)
    ww.setsampwidth(np.iinfo(dtype).bits / 8)
    ww.setframerate(fs)
    ww.writeframes(sig)
    ww.close()

def auroc(data1, data2, return_p=False):
    """Return auROC and two-sided p-value (if requested)"""
    try:
        U, p = scipy.stats.mannwhitneyu(data1, data2)
        p = p * 2
    except ValueError:
        print "some sort of error in MW"
        print data1
        print data2
        if return_p:
            return 0.5, 1.0
        else:
            return 0.5
    AUC  = 1 - (U / (len(data1) * len(data2)))

    if return_p:
        return AUC, p
    else:
        return AUC

def utest(x, y, return_auroc=False):
    """Drop-in replacement for scipy.stats.mannwhitneyu with different defaults
    
    If an error occurs in mannwhitneyu, this prints the message but returns
    a reasonable value: p=1.0, U=.5*len(x)*len(y), AUC=0
    
    This also calculates two-sided p-value and AUROC. The latter is only
    returned if return_auroc is True, so that the default is compatibility
    with mannwhitneyu.
    """
    badflag = False
    try:
        U, p = scipy.stats.mannwhitneyu(x, y)
        p = p * 2
    except ValueError as v:
        print "Caught exception:", v
        badflag = True

    # Calculate AUC
    if badflag:
        # Make up reasonable return values
        p = 1.0
        U = .5 * len(x) * len(y)
        AUC = .5
    elif len(x) == 0 or len(y) == 0:
        print "warning: one argument to mannwhitneyu is empty"
        AUC = .5
    else:
        AUC  = 1 - (U / (len(x) * len(y)))
    
    # Now return
    if return_auroc:
        return U, p, AUC
    else:
        return U, p

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

def times2bins_int(times, f_samp=1.0, t_start=None, t_stop=None):
    """Returns a 1-0 type spiketrain from list of times.
    
    Note the interface is different from times2bins, which is for
    general histogramming. This function expects times in seconds, and uses
    f_samp to convert to bins. The other function expects times in samples,
    and uses f_samp to convert to seconds.
    
    If multiple spikes occur in same bin, you still get 1 ... not sure
    this is right .... Essentially you're getting a boolean
    
    'times' : seconds
    'f_samp' : sampling rate of returned spike train
    """
    f_samp = float(f_samp)
    if t_stop is None:
        t_stop = times.max() + 1/f_samp
    if t_start is None:
        t_start = times.min()
    
    # set up return variable
    len_samples = np.rint(f_samp * (t_stop - t_start)).astype(np.int)
    res = np.zeros(len_samples, dtype=np.int)
    
    # set up times as indexes
    times = times - t_start
    times_samples = np.rint(times * f_samp).astype(np.int)
    times_samples = times_samples[~(
        (times_samples < 0) | (times_samples >= len(res)))]
    
    # set res
    res[times_samples] = 1
    return res
    
    

def times2bins(times, f_samp=None, t_start=None, t_stop=None, bins=10,
    return_t=False):
    """Given event times and sampling rate, convert to histogram representation.
    
    If times is list-of-list-like, will return list-of-list-like result.
    
    f_samp : This is for the case where `times` is in samples and you want
        a result in seconds. That is, times is divided by this value.
    
    Returns: res[, t_vals]
    Will begin at t_start and continue to t_stop
    """
    
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

def map_d(func, dic):
    """Map like func(val) for items in dic and maintain keys"""
    return dict([(key, func(val)) for key, val in dic.items()])

def filter_d(cond, dic):
    """Filter by cond(val) for items in dic and maintain keys"""
    return dict([(key, val) for key, val in dic.items() if cond(val)])
    

def getstarted():
    """Load all my data into kkpandas and RS objects
    
    Returns:
    xmlfiles, kksfiles, data_dirs, xml_roots, well_sorted_units, kk_servers
    
    Each is a dict keyed by ratname
    """
    xmlfiles = {
        'CR20B' : os.path.expanduser('~/Dropbox/lab/CR20B_summary/CR20B.xml'),
        'CR21A' : os.path.expanduser('~/Dropbox/lab/CR21A_summary/CR21A.xml'),
        'YT6A' : os.path.expanduser('~/Dropbox/lab/YT6A_summary/YT6A.xml'),
        }
    
    kksfiles = {
        'CR20B' : os.path.expanduser(
            '~/Dropbox/lab/CR20B_summary/CR20B_behaving.kks'),
        'CR21A' : os.path.expanduser(
            '~/Dropbox/lab/CR21A_summary/CR21A_behaving.kks'),
        'YT6A' : os.path.expanduser(
            '~/Dropbox/lab/YT6A_summary/YT6A_behaving.kks'),
        'CR17B' : os.path.expanduser(
            '~/Dropbox/lab/CR17B_summary/CR17B_behaving.kks'),
        'CR12B' : os.path.expanduser(
            '~/Dropbox/lab/CR12B_summary/CR12B_behaving.kks'),            
        }
    
    kk_servers = dict([
        (ratname, kkpandas.kkio.KK_Server.from_saved(kksfile))
        for ratname, kksfile in kksfiles.items()])
    
    data_dirs = {
        'CR20B' : '/media/hippocampus/chris/20120705_CR20B_allsessions',
        'CR21A' : '/media/hippocampus/chris/20120622_CR21A_allsessions',
        'YT6A' : '/media/hippocampus/chris/20120221_YT6A_allsessions',
        'CR17B' : '/media/hippocampus/chris/20110907_CR17B_allsessions',
        'CR12B' : '/media/hippocampus/chris/20111208_CR12B_allsessions_sorted',
        }
    
    xml_roots = dict([
        (ratname, etree.parse(xmlfile).getroot())
        for ratname, xmlfile in xmlfiles.items()])

    xpath_str = '//unit[quality/text()>=3 and ../../../@analyze="True"]'
    well_sorted_units = dict([
        (ratname, root.xpath(xpath_str))
        for ratname, root in xml_roots.items()])

    return xmlfiles, kksfiles, data_dirs, xml_roots, well_sorted_units, kk_servers

def load_channel_mat(filename, return_dataframe=True, dump_filler=True):
    """Load data from bao-lab format for single channel
    
    filename : path to file that has been broken out into each channel
        separately
    
    Each mat-file contains structured array like this:
    ans = 

                LFP: [1x763 single]
               PDec: [1x763 int16]
                 CH: [1x1 struct]
        Epoch_Value: [1 20 55]
    --- data.trial(1).CH
    ---- latency, spikewaveform    
    
    Returns: trials_info, spike_times
        trials_info : array of shape (n_trials, 3)
            if return_dataframe, this is a pandas.DataFrame instead
        spike_times : list of length n_trials, each containing array of
            trial-locked spike times.
    """
    # Load the structured array 'trial', with length n_trials
    data = scipy.io.loadmat(filename, squeeze_me=True)    
    trial = data['data']['trial'].item()
    
    # Here is the spikes from each trial
    # The purpose of the extra `flatten` is to ensure that arrays containing
    # 0 or 1 spike time are always 1-dimensional instead of 0-dimensional
    spike_times = map(lambda t: t.item()[2]['latency'].item().flatten(), trial) 

    # Here is the information about each trial
    trials_info = np.array(map(lambda t: t.item()[3], trial))
    
    # Remove trials containing nothing useful, ie delay 55ms or atten 70
    if dump_filler:
        bad_mask = (
            (trials_info[:, 2] == 55) |
            (trials_info[:, 1] == 70))
        trials_info = trials_info[~bad_mask]
        spike_times = list(np.asarray(spike_times)[~bad_mask])
    
    # Make a DataFrame
    if return_dataframe:
        trials_info = pandas.DataFrame(trials_info, 
            columns=['freq', 'atten', 'light'])    
    
    return trials_info, spike_times


def test_vs_baseline(data, baseline_idxs, test_idxs, debug_figure=False, 
    bins=10):
    """Test rows of a dataset against a baseline set of rows.
    
    For instance the rows could be timepoints and the columns replicates.
    Which timepoints are significantly greater (or less than) a set of
    timepoints defined as baseline?
    
    This tests each row in the test set separately against the full baseline
    test set using Mann-Whitney U and returns the p-value of each test row,
    along with some other diagnostics.
    
    One tricky part is determining the direction of the effect for the case
    of equal medians (for instance when the datasets, like spike counts, have
    median and mode 0). To know this for sure I would need to know small-u
    and big-u from within scipy.stats.mannwhitneyu. Instead I test the means,
    which is right most of the time but not all of the time.
    
    This behavior will probably change. I will calculate small-U and big-U
    separately and either return both or at least that to encode `dir`.
    
    Parameters
    ----------
    data : array-like, replicates in columns
    baseline_idxs : rows in data to be treated as baseline set
    test_idxs : row in data to test against baseline
    debug_figure : boolean, plot the histograms of all tested conditions
    bins : integer or array-like
        The bins to use when histogramming data. This only affects the
        debugging figure plot, and the returned histograms
    
    Returns
    -------
    u, p, auroc, dir, ref_counts, all_test_counts
    
    u : array of length N containing U statistic for each timepoint (row)
    p : array of length N containing two-tailed p-value for each row
    auroc : array of length N containg area under region of convergence
        for each row
    dir : boolean array of length N containing direction of effect
        if True, then the row exceeded the baseline. if False, it did not.
        Note this doesn't imply any significant difference.
        This is the part which currently comes (most of the time correctly)
        from comparing the means.
    ref_counts : histogram of baseline data
    all_test_counts : array of counts of test data, histogram bins on columns
    """
    # Check it's an array
    data = np.asarray(data)

    # get the baseline counts (note these may be non-integer due to smoothing)
    try:
        refdata = data[baseline_idxs].flatten()
    except IndexError:
        raise ValueError("baseline indexes are not valid row indexes")

    # for visualization/debugging, store histograms of all tests
    hmax = np.max([refdata.max(), data[test_idxs].max()])
    hmin = np.min([refdata.min(), data[test_idxs].min()])
    if not np.iterable(bins):
        bins = np.linspace(hmin, hmax, bins + 1)

    # histogram the reference set
    ref_counts = np.histogram(refdata, bins=bins)[0]

    # test each time bin separately
    all_test_counts, all_u, all_p, all_auroc, all_dir = [], [], [], [], []
    for idx in test_idxs:
        # get this test set
        try:
            testdata = data[idx].flatten()
        except IndexError:
            raise ValueError("provided test index not valid for provided data")
        
        # histogram it
        test_counts = np.histogram(testdata, bins=bins)[0]
        
        # test it
        u, p, auroc = utest(refdata, testdata, return_auroc=True)

        # Estimate direction of effect
        #dir = np.sum(testdata > np.median(np.concatenate([refdata, testdata]))) > \
        #    len(testdata) / 2.
        dir = np.mean(testdata) > np.mean(refdata)
        
        # store results
        all_test_counts.append(test_counts)
        all_u.append(u)
        all_p.append(p)
        all_auroc.append(auroc)
        all_dir.append(dir)

    # convert to array
    all_test_counts = np.asarray(all_test_counts) # row=timepoint, col=histbin
    all_u = np.asarray(all_u) # by timepoint
    all_p = np.asarray(all_p) # by timepoint
    all_auroc = np.asarray(all_auroc) # by timepoint
    all_dir = np.asarray(all_dir) # by timepoint

    # make test histograms into normalized cumulative
    if debug_figure:
        atc_toplot = np.cumsum(
            all_test_counts.astype(np.float) / all_test_counts.sum(axis=1)[:, None],
            axis=1)
        ref_toplot = np.cumsum(ref_counts.astype(np.float) / ref_counts.sum())

        f = plt.figure()
        for row, p, dir in zip(atc_toplot, all_p, all_dir):
            if p < .05 and dir:
                plt.plot(bins[:-1], row, 'r')
            elif p < .05:
                plt.plot(bins[:-1], row, 'g')
            else:
                plt.plot(bins[:-1], row, 'k')
        plt.plot(bins[:-1], ref_toplot, 'b', lw=2)

    return all_u, all_p, all_auroc, all_dir, ref_counts, all_test_counts
