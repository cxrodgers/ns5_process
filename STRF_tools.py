import numpy as np
import glob
import os.path
import matplotlib.mlab as mlab
import wave
import struct
import matplotlib.pyplot as plt
from myutils import Spectrogrammer
import myutils

def my_imshow(C, x=None, y=None, ax=None):
    if x is None:
        x = range(C.shape[1])
    if y is None:
        y = range(C.shape[0])
    extent = x[0], x[-1], y[0], y[-1]
    
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    
    plt.imshow(np.flipud(C), interpolation='nearest', extent=extent)
    ax.axis('auto')
    plt.show()

def clean_up_stimulus(whole_stimulus, silence_value='min_row', z_score=True):
    """Replaces non-finite values with silence_value, and z-scores by row.
    
    silence_value == 'min_row' : minimum value in the row
    silence_value == 'min_whole' : minimum value in the whole stimulus
    silence_value == 'mean_row', 'mean_whole' : mean
    silence_value == 'median_row', 'median_whole' : median
    
    You might want to check a histogram of the returned values and pick what
    looks best. On the one hand, silence is best represented by minimal
    power. On the other hand, this distorts the histogram due to silence
    in the actual signals, and/or the blanking periods.
    """
    if np.any(np.isnan(whole_stimulus)):
        print "warning: what to do with NaNs?"
    
    
    if silence_value is 'min_row':
        cleaned_stimulus = []
        for row in whole_stimulus:
            row[np.isneginf(row)] = row[~np.isneginf(row)].min()
            cleaned_stimulus.append(row)
        cleaned_stimulus_a = np.array(cleaned_stimulus)
    
    elif silence_value is 'mean_row':
        cleaned_stimulus = []
        for row in whole_stimulus:
            row[np.isneginf(row)] = row[~np.isneginf(row)].mean()
            cleaned_stimulus.append(row)
        cleaned_stimulus_a = np.array(cleaned_stimulus)
    
    elif silence_value is 'median_row':
        cleaned_stimulus = []
        for row in whole_stimulus:
            row[np.isneginf(row)] = np.median(row[~np.isneginf(row)])
            cleaned_stimulus.append(row)
        cleaned_stimulus_a = np.array(cleaned_stimulus)
    
    elif silence_value is 'min_whole':
        cleaned_stimulus_a = whole_stimulus.copy()
        cleaned_stimulus_a[np.isneginf(cleaned_stimulus_a)] = \
            np.min(cleaned_stimulus_a[np.isfinite(cleaned_stimulus_a)])
        
    elif silence_value is 'mean_whole':
        cleaned_stimulus_a = whole_stimulus.copy()
        cleaned_stimulus_a[np.isneginf(cleaned_stimulus_a)] = \
            np.mean(cleaned_stimulus_a[np.isfinite(cleaned_stimulus_a)])

    elif silence_value is 'median_whole':
        cleaned_stimulus_a = whole_stimulus.copy()
        cleaned_stimulus_a[np.isneginf(cleaned_stimulus_a)] = \
            np.median(cleaned_stimulus_a[np.isfinite(cleaned_stimulus_a)])
    
    else:
        # blindly assign 'silence_value' to the munged values
        cleaned_stimulus_a = whole_stimulus.copy()
        cleaned_stimulus_a[np.isneginf(cleaned_stimulus_a)] = silence_value
    
    if z_score:
        for n in range(cleaned_stimulus_a.shape[0]):
            s = np.std(cleaned_stimulus_a[n, :])
            if s < 10**-6: print "warning, std too small"
            cleaned_stimulus_a[n, :] = (
                cleaned_stimulus_a[n, :] - cleaned_stimulus_a[n, :].mean()) / \
                np.std(cleaned_stimulus_a[n, :])
    
    return cleaned_stimulus_a

class STRF_experiment:
    """Class that holds links to stimulus and response files"""
    stim_file_label = 'stim'
    spike_file_label = 'spike'
    stim_file_regex = r'stim(\d+)\.wav'
    spike_file_regex = r'spike(\d+)'
    
    def __init__(self, stim_dir=None, spike_dir=None):
        """Initialize object to hold stimulus and response data.
        
        stim_dir : directory holding files of name 'stim(\d+).wav', one
            wave file per trial, sampled at 200KHz.
        spike_dir : directory holding plaintext files of name 'spike(\d+)',
            one per trial, consisting of spike times separated by spaces
            in ms all on a single line, aligned to time zero the start of
            the corresponding wave file.
        """
        self.stim_dir = stim_dir
        self.spike_dir = spike_dir
        self.waveform_transformer = None
        self.error_check_filenames = True
        
        # list of files, to be set later automatically or by user
        self.wave_file_list = None
        self.spike_file_list = None

    def transform_all_stimuli(self, assert_sampling_rate=None, truncate=None):
        """Calculates spectrograms of each stimulus
        
        Saves in attributes `specgm_list`, `t_list`, and `freqs_list`.
        Also attempts to store unique `t` and `freq` for all.
        
        Returns spectrograms in (n_freqs, n_timepoints) shape
        """        
        # Get list of files to transform
        self._set_list_of_files()
        
        # where data goes
        self.specgm_list = []
        self.t_list = []
        self.freqs_list = []
        self.t = None
        self.freqs = None
        
        if self.waveform_transformer is None:
            # Use a Spectrogrammer with reasonable default parameters
            self.waveform_transformer = Spectrogrammer()
        
        # load and transform each file
        for wave_file in self.wave_file_list:
            waveform, fs = self.load_waveform_from_wave_file(wave_file)
            if assert_sampling_rate:
                assert fs == assert_sampling_rate
            
            specgm, freqs, t = self.waveform_transformer.transform(waveform)
            if truncate:
                inds = (t > truncate)
                t = t[~inds]
                specgm = specgm[:, ~inds]
            self.specgm_list.append(specgm)
            self.t_list.append(t)
            self.freqs_list.append(freqs)
                
        # Store unique values of t and freqs (if possible)
        if len(self.t_list) > 0 and np.all(
            [np.all(tt == self.t_list[0]) for tt in self.t_list]):
            self.t = self.t_list[0]
        if len(self.freqs_list) > 0 and np.all(
            [ff == self.freqs_list[0] for ff in self.freqs_list]):
            self.freqs = self.freqs_list[0]
    
    def load_waveform_from_wave_file(self, filename, dtype=np.float):
        """Opens wave file and reads, assuming signed shorts"""
        wr = wave.Wave_read(filename)
        fs = wr.getframerate()
        sig = np.array(struct.unpack('%dh' % wr.getnframes(), 
            wr.readframes(wr.getnframes())), dtype=dtype)
        wr.close()
        return sig, fs
    
    def _set_list_of_files(self):
        """Reads stimulus and response filenames from disk, if necessary.
        
        If the attributes are already set, do not reload from disk (so
        you can overload this behavior).
        
        In any case, error check that the lists are the same length and end
        with the same sequence of digits, eg [spike003, spike007] and
        [stim003, stim007].
        """
        if self.wave_file_list is None:
            # Find sorted list of wave files
            self.wave_file_list = sorted(glob.glob(os.path.join(self.stim_dir,
                self.stim_file_label + '*.wav')))
        
        if self.spike_file_list is None:
            # Find sorted list of spike files
            self.spike_file_list = sorted(glob.glob(os.path.join(self.spike_dir,
                self.spike_file_label + '*')))
                
        # Error checking
        if self.error_check_filenames:
            assert len(self.spike_file_list) == len(self.wave_file_list)
            for wave_file, spike_file in zip(self.wave_file_list, 
                self.spike_file_list):
                # extract numbers on end of wave and spike files
                wave_num = glob.re.search(self.stim_file_regex,
                    wave_file).groups()[0]
                spike_num = glob.re.search(self.spike_file_regex,
                    spike_file).groups()[0]
                
                # test string equality (3 != 003)
                assert wave_num == spike_num
    
    def get_full_stimulus_matrix(self, n_delays, blanking_value=-np.inf):
        """Concatenate and reshape spectrograms for STRF estimation.
        
        Returned value is one large spectogram of shape (n_freqs*n_delays, 
        total_time). Each column contains the current value and the previous
        n_delays.    
        
        Puts (n_freqs, n_delays) blanking_values before each stimulus.
        Then reshapes each column to include delays:
            column_n = concatenated_specgram[:, n:n-n_delays:-1].flatten()
        
        The original slice corresponding to column n:
            concatenated_specgram[:, n:n-n_delays:-1]
        can be recovered as:
            reshaped_specgram[:, n].reshape(specgram.shape[0], n_delays)
        
        There are n_delays blanks in between each stimuli, but the total length
        of the returned value is the sum of the total length of the provided
        stimuli because those blanks are folded into the features. That is, the
        first entry contains the first time sample and the rest blanks; and
        the last sample contains the last n_delays samples.
        """   
        if len(self.specgm_list) == 0:
            print "nothing to concatenate, have you run transform_all_stimuli?"
            return
        
        # put blanks in front of each stimulus and concatenate
        concatenated_specgram_list = []
        for specgram in self.specgm_list:
            # first prepend blanks
            specgram_with_prepended_blanks = np.concatenate([
                blanking_value * np.ones((specgram.shape[0], n_delays)), 
                specgram], axis=1)
            
            # now reshape and include delays in each feature
            reshaped_specgram_list = []
            for n in range(n_delays, specgram_with_prepended_blanks.shape[1]):
                reshaped_specgram_list.append(
                    specgram_with_prepended_blanks[:, n:n-n_delays:-1].flatten())
            reshaped_specgram = np.transpose(np.array(reshaped_specgram_list))
            
            concatenated_specgram_list.append(reshaped_specgram)
        concatenated_specgram = np.concatenate(concatenated_specgram_list, axis=1)
        
        return concatenated_specgram        
    
    def get_concatenated_stimulus_matrix(self):
        """Returns a concatenated (non-reshaped) matrix of stimuli.
        
        I think this is just np.concatenate(self.specgm_list, axis=?)
        """
        pass
    
    def get_concatenated_response_matrix(self, dtype=np.float, 
        sampling_rate=1000., truncate=None):    
        """Loads spike files from disk, returns concatenated responses.
        
        You must run transform_all_stimuli first, or otherwise set self.t_list,
        so that I know how to bin the spikes.
        
        Returns in shape (1, N_timepoints)
        """        
        # Set list of filenames and error check
        self._set_list_of_files()
        
        # load each one and histogram
        concatenated_psths = []
        for respfile, bin_centers in zip(self.spike_file_list, self.t_list):
            # store responses
            #~ try:
                #~ # flatten() handles the case of only one value
                #~ st = np.loadtxt(respfile).flatten()
            #~ except IOError:
                #~ # this handles the case of no data
                #~ st = np.array([])
            #~ st = st / 1000.0
            s = file(respfile).readlines()
            st = []
            for line in s:
                tmp = myutils.parse_space_sep(line, dtype=np.float)
                tmp = np.asarray(tmp) / sampling_rate
                if truncate:
                    tmp = tmp[tmp <= truncate]
                st.append(tmp)
            
            # convert bin centers to bin edges
            bin_edges = bin_centers[:-1] + 0.5 * np.diff(bin_centers)
            bin_edges = np.concatenate([[-np.inf], bin_edges, [np.inf]])
            
            # now histogram
            counts = []
            for line in st:
                counts.append(np.histogram(line, bin_edges)[0])
            counts = np.mean(counts, axis=0)
        
            # Append to growing list and check that size matches up trial-by-trial
            concatenated_psths.append(counts)
            assert len(counts) == len(bin_centers)
        
        # Return a concatenated array of response from this recording
        return np.concatenate(concatenated_psths).astype(dtype)[np.newaxis,:]


#~ class Spectrogrammer:
    #~ """Turns a waveform into a spectrogram"""
    #~ def __init__(self, NFFT=256, downsample_ratio=5, new_bin_width_sec=None,
        #~ max_freq=40e3, min_freq=5e3, Fs=200e3, normalization=1.0):
        #~ """Initialize object to turn waveforms to spectrograms.
        
        #~ Stores parameter choices, so you can batch analyze waveforms using
        #~ the `transform` method.
        
        #~ If you specify new_bin_width_sec, this chooses the closest integer 
        #~ downsample_ratio and that parameter is actually saved and used.
        
        #~ TODO: catch other kwargs and pass to specgram.
        #~ """
        
        #~ # figure out downsample_ratio
        #~ if new_bin_width_sec is not None:
            #~ self.downsample_ratio = int(np.rint(new_bin_width_sec * Fs / NFFT))
        #~ else:
            #~ self.downsample_ratio = int(downsample_ratio)
        #~ assert self.downsample_ratio > 0
        
        #~ # store other defaults
        #~ self.NFFT = NFFT
        #~ self.max_freq = max_freq
        #~ self.min_freq = min_freq
        #~ self.Fs = Fs
        #~ self.normalization = normalization

    
    #~ def transform(self, waveform):
        #~ """Converts a waveform to a suitable spectrogram.
        
        #~ Removes high and low frequencies, rebins in time (via median)
        #~ to reduce data size. Returned times are the midpoints of the new bins.
        
        #~ Returns:  Pxx, freqs, t    
        #~ Pxx is an array of dB power of the shape (len(freqs), len(t)).
        #~ It will be real but may contain -infs due to log10
        #~ """
        #~ # For now use NFFT of 256 to get appropriately wide freq bands, then
        #~ # downsample in time
        #~ Pxx, freqs, t = mlab.specgram(waveform, NFFT=self.NFFT, Fs=self.Fs)
        #~ Pxx = Pxx * np.tile(freqs ** self.normalization, (1, Pxx.shape[1]))

        #~ # strip out unused frequencies
        #~ Pxx = Pxx[(freqs < self.max_freq) & (freqs > self.min_freq), :]
        #~ freqs = freqs[(freqs < self.max_freq) & (freqs > self.min_freq)]

        #~ # Rebin in size "downsample_ratio". If last bin is not full, discard.
        #~ Pxx_rebinned = []
        #~ t_rebinned = []
        #~ for n in range(0, len(t) - self.downsample_ratio + 1, 
            #~ self.downsample_ratio):
            #~ Pxx_rebinned.append(
                #~ np.median(Pxx[:, n:n+self.downsample_ratio], axis=1).flatten())
            #~ t_rebinned.append(
                #~ np.mean(t[n:n+self.downsample_ratio]))

        #~ # Convert to arrays
        #~ Pxx_rebinned_a = np.transpose(np.array(Pxx_rebinned))
        #~ t_rebinned_a = np.array(t_rebinned)

        #~ # log it and deal with infs
        #~ Pxx_rebinned_a_log = -np.inf * np.ones_like(Pxx_rebinned_a)
        #~ Pxx_rebinned_a_log[np.nonzero(Pxx_rebinned_a)] = \
            #~ 10 * np.log10(Pxx_rebinned_a[np.nonzero(Pxx_rebinned_a)])

        #~ return Pxx_rebinned_a_log, freqs, t_rebinned_a

def clean_up_stimulus(whole_stimulus, silence_value='min_row', z_score=True):
    """Replaces non-finite values with silence_value, and z-scores by row.
    
    silence_value == 'min_row' : minimum value in the row
    silence_value == 'min_whole' : minimum value in the whole stimulus
    silence_value == 'mean_row', 'mean_whole' : mean
    silence_value == 'median_row', 'median_whole' : median
    
    You might want to check a histogram of the returned values and pick what
    looks best. On the one hand, silence is best represented by minimal
    power. On the other hand, this distorts the histogram due to silence
    in the actual signals, and/or the blanking periods. It also means that
    this silence will play a part in the linear fit.
    
    Most models treat the rows independently, so if you set the silence
    row-independently, it will mesh nicely. On the other hand, why should
    silence in one frequency band be treated differently from others?
    """
    #~ if np.any(np.isnan(whole_stimulus)):
        #~ print "warning: what to do with NaNs?"
    
    
    if silence_value is 'min_row':
        cleaned_stimulus_a = whole_stimulus.copy()
        for n in range(len(cleaned_stimulus_a)):
            msk = np.isneginf(cleaned_stimulus_a[n])
            cleaned_stimulus_a[n, msk] = cleaned_stimulus_a[n, ~msk].min()
    
    elif silence_value is 'mean_row':
        cleaned_stimulus = []
        for row in whole_stimulus:
            row[np.isneginf(row)] = row[~np.isneginf(row)].mean()
            cleaned_stimulus.append(row)
        cleaned_stimulus_a = np.array(cleaned_stimulus)
    
    elif silence_value is 'median_row':
        cleaned_stimulus = []
        for row in whole_stimulus:
            row[np.isneginf(row)] = np.median(row[~np.isneginf(row)])
            cleaned_stimulus.append(row)
        cleaned_stimulus_a = np.array(cleaned_stimulus)
    
    elif silence_value is 'min_whole':
        cleaned_stimulus_a = whole_stimulus.copy()
        cleaned_stimulus_a[np.isneginf(cleaned_stimulus_a)] = \
            np.min(cleaned_stimulus_a[np.isfinite(cleaned_stimulus_a)])
        
    elif silence_value is 'mean_whole':
        cleaned_stimulus_a = whole_stimulus.copy()
        cleaned_stimulus_a[np.isneginf(cleaned_stimulus_a)] = \
            np.mean(cleaned_stimulus_a[np.isfinite(cleaned_stimulus_a)])

    elif silence_value is 'median_whole':
        cleaned_stimulus_a = whole_stimulus.copy()
        cleaned_stimulus_a[np.isneginf(cleaned_stimulus_a)] = \
            np.median(cleaned_stimulus_a[np.isfinite(cleaned_stimulus_a)])
    
    else:
        # blindly assign 'silence_value' to the munged values
        cleaned_stimulus_a = whole_stimulus.copy()
        cleaned_stimulus_a[np.isneginf(cleaned_stimulus_a)] = silence_value
    
    if z_score:
        for n in range(cleaned_stimulus_a.shape[0]):
            s = np.std(cleaned_stimulus_a[n, :])
            if s < 10**-6: print "warning, std too small"
            cleaned_stimulus_a[n, :] = (
                cleaned_stimulus_a[n, :] - cleaned_stimulus_a[n, :].mean()) / \
                np.std(cleaned_stimulus_a[n, :])
    
    return cleaned_stimulus_a


class DirectFitter:
    """Calculates STRF for response matrix and stimulus matrix"""
    def __init__(self, X=None, Y=None):
        """New fitter
        X : (n_timepoints, n_features).
            I think this works better if mean of rows and cols is zero.
        Y : (n_timepoints, 1)
            Generally 0s and 1s.
        """
        self.X = X
        self.Y = Y
        self.XTX = None # pre-calculate this one
        
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("n_timepoints (dim0) is not the same!")
        if self.X.shape[1] > self.X.shape[1]:
            print "warning: more features than timepoints, possibly transposed"
    
    def STA(self):
        """Returns spike-triggered average of stimulus X and response Y.
        
        Therefore the STA is given by np.dot(X.transpose(), Y) / Y.sum().
        """
        return np.dot(self.X.transpose(), self.Y).astype(np.float) / \
            self.Y.sum()
        #X = X - X.mean(axis=0)[newaxis, :] # each feature has zero-mean over time
        #X = X - X.mean(axis=1)[:, newaxis] # each datapoint has zero-mean over features

    def whitened_STA(self, ridge_parameter=0.):
        if self.XTX is None:
            self.XTX = np.dot(self.X.transpose(), self.X)
        
        ridge_mat = ridge_parameter * len(self.XTX)**2 * np.eye(len(self.XTX))
        STA = self.STA()
    
        return np.dot(np.linalg.inv(self.XTX + ridge_mat), STA)*self.X.shape[0]

