"""Module providing objects to detect audio onsets."""

import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import sys

class OnsetDetector(object):
    """Given a mono or stereo audio stream, detects sound onsets.
    
    In every case, the audio stream is squared and smoothed to
    calculate `smoothed_power`. When smoothed_power crosses a
    threshhold, an onset is marked.
    
    Later error checking removes spurious onsets.
    """
    def __init__(self, input_data, F_SAMP=30000., manual_threshhold=None,
        minimum_threshhold=None, minimum_duration_ms=5, 
        plot_debugging_figures=False, verbose=False):
        """An object that intelligently detects audio onsets in its input.
        
        Parameters
        ----------
        input_data: 1d or 2d array of values. If 2d, the threshhold crossing
            code is run on each channel separately. The waveform need
            only cross threshhold on one channel, not both.
        
        manual_threshhold: Specify a threshhold manually. Events will be
            detected when the smoothed audio power crosses this threshhold.
            Thus, make sure that you specify in units**2.
            (Regular units, not dB.)
        
        minimum_duration_ms: Sounds less than this duration are discarded
        
        
        Call method execute() to actually run the code.
        
        If manual_threshhold is None, a threshhold will be chosen for you.
        In the case of stereo data, the threshhold is set using the first
        channel (usually left) and then the same threshhold is used for
        the other channel. The threshold will be stored in self.threshold
        
        Implementation note: slowest part of execution appears
        to be the smoothing filter. Look into implementing more efficiently.
        Smoothing an array of length 2**24 (17M) is barely tolerable (10sec).
        """
        
        self.input_data = input_data
        self.threshhold = manual_threshhold
        self._minimum_threshhold = minimum_threshhold
        self.F_SAMP = F_SAMP
        self._minimum_duration_samples = \
            np.rint(minimum_duration_ms / 1000. * self.F_SAMP)
        self.plot_debugging_figures = plot_debugging_figures
        self.detected_onsets = None # so far
        self.verbose = verbose
        
        
        # We use a causal filter to be extra sure that we don't err on the side
        # of identifying the onset too soon. Note that this guarantees we will
        # identify the onset too late! Well, actually, since the threshhold
        # is set based on the delayed data, it doesn't matter too much.
        # Check debugging figure to reassure yourself that the delay is not
        # significant.
        flen = np.rint(.003*self.F_SAMP) # 3ms
        self.smoother = CausalRectangularSmoother(smoothing_filter_length=flen)
        
        
        # This is the best onset detector so far, out of the implementations
        # I've tried. It will only be used if manual threshhold not
        # specified.
        self.thresh_setter = ThreshholdAutosetterLeastSensitive
        
        # Check that an impulse won't be so smeared by the smoother that
        # it would pass the minimum duration requirement.
        if self._minimum_duration_samples < \
            self.smoother.smoothing_filter_length:
            print "WARNING: with current smoothing settings, even an " \
                "impulse could pass the min_duration requirement!"
    
    
    def execute(self):
        """Executes onset detection.
        
        If onsets or offsets are within smoothing filter length of
        beginning or end, a warning is generated.
        """        
        
        # Deal with the case of stereo input
        if self.input_data.ndim == 2:
            sound_bool = self._find_stereo_threshhold_crossings(self.input_data)
        elif self.input_data.ndim == 1:
            sound_bool = self._find_mono_threshhold_crossings(self.input_data)                

        # We need to throw out sounds that are too short.
        # Finally, will store self.detected_onsets.
        if self.verbose:
            print "Error checking"; sys.stdout.flush()
        self._error_check_onsets(sound_bool)
        
        # Check for sounds too close to beginning
        if len(self.detected_onsets >= 1):
            if self.detected_onsets[0] < self.smoother.smoothing_filter_length:
                print "warning: first onset is within filter width of beginning"
            if self.detected_offsets[-1] > \
                (len(sound_bool) - self.smoother.smoothing_filter_length):
                print "warning: last offset is within filter width of end"
    
    def _find_mono_threshhold_crossings(self, data_vector):
        """Finds threshhold crossings in mono data_vector.
        
        If manual threshhold is None, attempts to autoset.
        Returns a boolean array which is True when the signal
        exceeds threshhold.
        
        TODO: move this method and _find_stereo to base class
        ThreshholdSetter. Doesn't need to know about anything else!
        
        Rearrange stuff a little so that smoothing can be done in
        parallel with multiple processes.
        """

        # Remove mean
        data_vector = data_vector - data_vector.mean()
        
        # Appears to work best when smoothing power first, then putting
        # into dB.
        # smooth the audio data     
        if self.verbose:
            print "Smoothing the data"; sys.stdout.flush()
        smoothed_power_dB = 20*np.log10(\
            self.smoother.execute(data_vector**2))
        
        # Comment to save memory, uncomment for debugging
        self.smoothed_power_dB = smoothed_power_dB
        
        # Autoset threshhold if it wasn't set already
        if self.threshhold is None:
            if self.verbose:
                print "Autosetting threshhold"; sys.stdout.flush()

            # Instantiate and run threshhold setter
            self.tset = self.thresh_setter(\
                input_data=smoothed_power_dB,
                plot_debugging_figures=self.plot_debugging_figures)
            th = self.tset.execute()
            
            # Apply minimum threshhold test
            if th > self._minimum_threshhold:
                self.threshhold = th
            else:
                self.threshhold = self._minimum_threshhold        
        
        # find when smoothed waveform exceeds threshhold
        if self.verbose:
            print "Finding threshhold crossings at %0.3f dB" % self.threshhold
            sys.stdout.flush()
        
        return smoothed_power_dB > self.threshhold
    
    
    def _find_stereo_threshhold_crossings(self, data_vector):
        """Finds threshhold crossings in stereo data_vector
        
        We know self.input_data.ndim == 2. Calls the mono detector once on 
        left channel and once again on right channel. Note that the left
        channel threshhold will be used for the right. If this is not
        appropriate, considering setting the threshhold manually.
        
        Returns a boolean array of same shape as one row of self.input_data
        which is True whenever either of the channels is above threshhold.
        """
        
        if data_vector.shape[0] != 2:
            print "WARNING: audio data should have shape (2,N)."
            data_vector = data_vector.transpose()

        sound_bool_L = self._find_mono_threshhold_crossings(data_vector[0])
        sound_bool_R = self._find_mono_threshhold_crossings(data_vector[1])
        
        return sound_bool_L | sound_bool_R
    
    
    def _error_check_onsets(self, sound_bool):
        # Find when the threshhold crossings first happen. `onsets` and
        # `offsets` are inclusive bounds on audio power above threshhold.
        onsets = mlab.find(np.diff(np.where(sound_bool, 1, 0)) == 1) + 1
        offsets = mlab.find(np.diff(np.where(sound_bool, 1, 0)) == -1)
        
        # check that we don't start or end in the middle of a sound
        try:
            if onsets[0] > offsets[0]:
                # Extra offset at the beginning
                offsets = offsets[1:]
            if onsets[-1] > offsets[-1]:
                # Extra onset at the end
                onsets = onsets[:-1]
        except IndexError:
            # apparently no onsets or no offsets
            print "No sounds found!"
            onsets = np.array([])
            offsets = np.array([])            
        
        
        if len(onsets) > 0:
            # First do some error checking.
            assert (len(onsets) == len(offsets)) and (np.all(onsets <= offsets))
                    
            # Remove sounds that violate min_duration requirement.
            too_short_sounds = ((offsets - onsets) < self._minimum_duration_samples)
            if np.any(too_short_sounds):
                if self.verbose:
                    print "Removing %d sounds that violate duration requirement" % \
                        len(mlab.find(too_short_sounds))
                
            onsets = onsets[np.logical_not(too_short_sounds)]
            offsets = offsets[np.logical_not(too_short_sounds)]
            
            # Warn when onsets occur very close together. This might occur if the 
            # sound power briefly drops below threshhold.
            if np.any(np.diff(onsets) < self._minimum_duration_samples):                
                print "WARNING: %d onsets were suspiciously close together." % \
                    len(find(np.diff(onsets) < self._minimum_duration_samples))
            
            # Print the total number of sounds identified.
            if self.verbose:
                print "Identified %d sounds with average duration %0.3fs" % \
                    (len(onsets), (offsets-onsets).mean() / self.F_SAMP)
        
        
        # Store detected onsets
        self.detected_onsets = onsets
        self.detected_offsets = offsets
    
    
    def _plot_debugging_figure(self, smoothed_audio_power, 
        win_duration_ms=5):
        # debugging figure
        # Plot all sound waveforms overlaid, to verify that they were caught
        # correctly.
        # This parameter determines how large the plotting window is.
        WINDOW_HALF_DURATION = np.rint(win_duration_ms/1000.0*self.F_SAMP) # samples
        
        # Initialize the figure and subplots.        
        f = plt.figure()
        ax = [f.add_subplot(2,2,n+1) for n in xrange(4)]            
        ax[0].set_title('Onset of sounds')
        ax[1].set_title('Offset of sounds')
        ax[2].set_title('Onset of smoothed')
        ax[3].set_title('Offset of smoothed')
            
        # Plot threshholds on the smoothed plots
        ax[2].plot([-WINDOW_HALF_DURATION, WINDOW_HALF_DURATION],\
            self.threshhold * np.ones((2,)), 'k:')
        ax[3].plot([-WINDOW_HALF_DURATION, WINDOW_HALF_DURATION],
            self.threshhold * np.ones((2,)), 'k:')
        
        # Now plot close-up of onset and offset for each sound
        for onset, offset in zip(self.detected_onsets, 
            self.detected_offsets):                
            # Deal with case where close to edge of window. Set new
            # boundaries that are within the data range. These boundaries
            # are used below in the plotting commands.
            start_win = max(onset - WINDOW_HALF_DURATION, 0)
            stop_win = min(offset + WINDOW_HALF_DURATION, 
                len(self.input_data))
            
            # Plot a close-up of the onset
            ax[0].plot(\
                np.arange(start_win-onset, WINDOW_HALF_DURATION),
                self.input_data[start_win:onset+WINDOW_HALF_DURATION])
            
            # Plot a close-up of the offset
            ax[1].plot(\
                np.arange(-WINDOW_HALF_DURATION, stop_win - offset),
                self.input_data[offset-WINDOW_HALF_DURATION:stop_win])
            
            # Now do the same but with the smoothed data
            ax[2].plot(\
                np.arange(start_win - onset, WINDOW_HALF_DURATION),
                smoothed_audio_power[start_win:onset+WINDOW_HALF_DURATION])
            
            ax[3].plot(\
                np.arange(-WINDOW_HALF_DURATION, stop_win - offset),
                smoothed_audio_power[offset-WINDOW_HALF_DURATION:stop_win])
        
        plt.show()

    
    def commit_audio_onsets(self):
        """Writes the newly detected audio onsets to disk.
        
        """
        # Note: even on a 32-bit system, this format allows >19hrs of indices.
        np.savetxt('audio_onsets', self.detected_onsets, '%i')
        #self.raw_data_loader.audio_onsets = self.audio_onsets



    

class CausalRectangularSmoother(object):
    def __init__(self, smoothing_filter_length=100):
        self.smoothing_filter_length = smoothing_filter_length
        self.smoothing_filter = None
        self.build_filter()
    
    
    def build_filter(self):
        """Rectangular filter of unity gain"""
        fillval = 1. / self.smoothing_filter_length
        fillshape = (self.smoothing_filter_length,)
        
        self.smoothing_filter_b = np.ones(fillshape) * fillval # numerator
        self.smoothing_filter_a = np.array([1]) # denominator
        self.filtering_function = sp.signal.lfilter # causal, 1d data only
    
    
    def execute(self, input_data):
        smoothed = self.filtering_function(b=self.smoothing_filter_b,
            a=self.smoothing_filter_a, x=input_data)
        return smoothed


class ThreshholdAutosetter(object):
    """Given an unordered stream of data, chooses an event threshhold.
    
    Different algorithms are possible. Each operates on the distribution
    of data points to choose an "intelligent" threshhold, ie, one in which
    only very large events of non-negligible duration cross.
    """        
    def __init__(self, input_data, min_p_events=.0001, max_p_events=.99,
        max_data_points=50e6, plot_debugging_figures=False):
        """Initialize a threshhold setter.
        
        Parameters
        ----------
        input_data: 1d array of data. The order is irrelevant.
            Generally you will want to provide this in dB, eg
            20*np.log10(data), so that it "looks linear".
        min_p_events: minimum fraction of the data points that will be
            above threshhold.
        max_p_events: maximum fraction of the data points that will be
            above threshhold.
        max_data_points: To increase performance, we do not to analyze
            all of the data points. If max_data_points<len(input_data),
            input_data will be strided to keep only max_data_points.
        """
        self.min_p_events = min_p_events
        self.max_p_events = max_p_events
        self.plot_debugging_figures = plot_debugging_figures
        
        if input_data.size > max_data_points:
            stride = np.ceil(input_data.size / max_data_points)
            self.input_data = input_data[::stride]
            #print "old size %d, new size %d" % (input_data.size, 
            #    self.input_data.size)
        else:
            self.input_data = input_data
    

class ThreshholdAutosetterLeastSensitive(ThreshholdAutosetter):
    def execute(self):
        """Automatically calculates a reasonable threshhold to detect onsets.
        
        Theory
        ------
        First calculates the distribution of power in the audio signal
        across time. Presumably, for the great majority of the time,
        this power will be low. Occasionally, there will be infrequent
        bursts of audio power, which are the stimuli to be detected.
        
        We want to set the threshhold to split these two regimes. This
        algorithm first limits the search to threshholds that satisfy the
        minimum and maximum event probabilities set in `min_p_events`
        and `max_p_events`. Within that regime, it finds the largest gap
        between observed data samples, and sets the threshhold there.
        """
        x = np.sort(self.input_data)        
        
        # Apply constraints
        search_regime = \
            (np.arange(len(x)) > self.min_p_events*len(x)) & \
            (np.arange(len(x)) < self.max_p_events*len(x))
        
        # Find largest gap in that regime
        diff_x_in_regime = np.diff(x[search_regime])
        idx_largest_diff = np.argmax(diff_x_in_regime)
        
        # Set threshhold in middle of that gap
        best_thresh = 0.5 * (x[search_regime][idx_largest_diff] + \
            x[search_regime][idx_largest_diff + 1])
        return best_thresh


class ThreshholdAutosetterMinimalHistogram(ThreshholdAutosetter):
    def execute(self, nbins=500):        
        """Automatically calculates a reasonable threshhold to detect onsets.
        
        Theory
        ------
        First calculates the distribution of power in the audio signal
        across time. Presumably, for the great majority of the time,
        this power will be low. Occasionally, there will be infrequent
        bursts of audio power, which are the stimuli to be detected.
        
        We want to set the threshhold to split these two regimes. This
        algorithm takes a weighted average of all possible power threshholds,
        weighting powers that rarely occur in the data more highly.
        
        Only threshholds that satisfy the `min_p_events` and `max_p_events`
        criteria are considered. Thus, you are guaranteed that the fraction
        of events exceeding the threshhold will be between these two
        values.
        
        
        Parameters
        ----------
        nbins: The distribution is first binned into this many bins to
            make analysis easier. Default: 500
        """
        # First bin the input data
        # This data should `look linear`. The user should convert power
        # to dB before passing to this function, for example.        
        (h, bins) = np.histogram(self.input_data, bins=nbins)
        
        # Convert to bin centers
        bins = bins[:-1] + 0.5*np.diff(bins)
        
        # Debugging figure
        if self.plot_debugging_figures:
            plt.figure(); plt.plot(bins, h);
            plt.title('Histogram of input data')
            plt.xlabel('input values'); plt.ylabel('frequency')
            plt.show()            
        
        # Calculate the part of the histogram that satisfies the minimum
        # and maximum constraints.
        cumhist = np.float64(np.cumsum(h)) / len(self.input_data)
        search_regime = \
            (cumhist > self.min_p_events) & \
            (cumhist < self.max_p_events)
            
        # Apply those constraints to `bins` and `h`
        bins = bins[search_regime]
        h = np.float64(h[search_regime])
        if self.plot_debugging_figures:
            plt.plot(bins[0], 0, 'r*')
            plt.plot(bins[-1], 0, 'r*')
            plt.show()
        
        # Lower points in the histogram make better threshholds, so invert
        h = np.max(h) - h
        
        # Now a weighted average gives the index of the best threshhold. The
        # sparsest bins are weighted the most.
        h = h / np.sum(h)
        weighted_best_bin = np.sum(h * np.arange(len(h)))
        best_thresh = bins[int(round(weighted_best_bin))]
        if self.plot_debugging_figures:
            plt.plot(best_thresh, 0, 'k*')
            plt.show()
        
        return best_thresh



