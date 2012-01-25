import numpy as np
import pdb
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import sys
import os

class DataSession:
    """A container to load and prepare data from a single experiment.
    
    """
    def __init__(self, dsdir='.', plot_debugging_figures=False):
        """Initialize a new data session.
        
        Parameters
        ----------
        dsdir : string, optional
            The directory to store all computed information from this
            object. (TODO: implement this!)
            Default is '.'
            KNOWN BUG: currently only works if you use '.', the current
            working directory! So please do not specify any other
            parameter. This is first on my list of things to fix.
        
        plot_debugging_figures : optional, boolean
            If True, display several figures with matplotlib throughout
            the operation of the DataSession. These figures are not
            critical and are generally only used when debugging.
            Default is True
            (TODO: figure out why this causes large memory leak!)
        
        
        Attributes
        ----------
        raw_data_loader : Loader-like object, optional
            The source for neural and audio data for this session.
            Assign this attribute.
        
        bcontrol_loader : Loader-like object, optional
            The source for behavioral data for this session.
            Assign this attribute.
        
        stim_syncer : Instance of Syncer
            Object that knows how to sync the data sources.
            Assign this attribute.
        
        
        Methods
        -------
        sync
            Calls the associated stim_syncer with the parameters it
            needs to sync the data sources contained by this
            DataSession. Currently this is always bcontrol_loader
            and raw_data_loader.
        
        extract_audio_onsets
            Loads audio data from raw_data_loader and searches for
            audio stimuli. The onsets of these stimuli are stored.
            TODO: make one for bcontrol_loader too!
            TODO: could be LazyObject
        
        commit_audio_onsets
            Store the recently calculated audio onsets.
            TODO: Make this _commit_audio_onsets?
        
        autocalc_threshhold
            Calculates an audio threshhold for determining an onset.
            TODO: Make this _autocalc_threshhold?
        
        
        Notes
        -----
        So you have a day of data and you want to analyze it?
        Generally you will want to follow these steps:
            1)  Initialize a new empty DataSession `ds`
            2)  Create a DataLoader object and add it to `ds`, for
                instance raw_data_loader
            3)  Create a DataLoader object and add it to `ds`, for
                instance bcontrol_loader
            4)  Call computations on these data, for instance
                `extract_audio_onsets` and `sync`
            5)  Create a viewer for the data and pass `ds` to the viewer
            
            See following example script:
            
            ds = DataSession.DataSession(dsdir='./mydatasession/)
            ds.raw_data_loader = RawDataLoader.ns5_Loader(settings='HO4')
            ds.raw_data_loader.load_file('/path/to/rawdata')
            ds.bcontrol_loader = RawDataLoader.BControlLoader(\
                '/path/to/behaviordata)
            ds.bcontrol_loader.load_file()
            ds.extract_audio_events()
            #ds.extract_spike_times() # Not yet implemented
            ds.stim_syncer = DataSession.BehavingSyncer()
            ds.sync()
            lfp_viewer = EventTriggeredAverage.LFPViewer()
            lfp_viewer.calculate(ds)
        """
        self.dsdir = dsdir;
        self.audio_onsets = None        
        self.plot_debugging_figures = plot_debugging_figures
        self.smoothing_filter_length = 100
    

    def sync(self):
        """Calls my contained stim_syncer with the info it needs to do its job.
        
        """
        self.syncing_function = self.stim_syncer.sync(self.bcontrol_loader,
            self.raw_data_loader)
    
    
    def extract_audio_events(self, min_duration_ms=10, manual_threshhold=None,
        minimum_threshhold=None):
        """Extracts the onset times of events in the audio stream.
        
        Looks for events in the audio stream that exceed a power threshhold,
        which can be manually specified or automatically determined.
        """
        
        # First check to see whether the audio onsets were already calculated.
        # TODO: What if self.audio_onsets is not None?
        # TODO: Also check whether parameters have changed, which should trigger
        # a recalculation.
        try:
            self.audio_onsets = np.loadtxt('audio_onsets', dtype=np.int64)
            self.raw_data_loader.audio_onsets = self.audio_onsets
            print "Loaded %d previously calculated audio onsets." % \
                len(self.audio_onsets)
            return
        except IOError as inst:
            if inst[0] == 'End-of-file reached before encountering data.':
                # File exists, but contains no data
                print "Loaded blank audio_onsets, no sounds found."
                self.audio_onsets = np.array([])
                return
            elif inst[0] == 2:
                # No such file exists. Let's calculate the audio onsets.
                "Calculating audio onsets..."
            else:
                print "Unexpected exception: {0}".format(str(inst))
                raise
        
        
        min_duration_samples = min_duration_ms / 1000. / \
            (self.raw_data_loader.header.period / 30000.)
        if min_duration_samples < 2*self.smoothing_filter_length:
            print "WARNING: with current smoothing settings, even an " \
                "impulse could pass the min_duration requirement!"
                

        # load into memory
        # This type of line will take forever to complete, just slice it!
        # audio_data = np.array(self.raw_data_loader.audio_data)
        audio_data = self.raw_data_loader.audio_data[:,:]
                                
        # smooth the audio data               
        # We use a causal filter to be extra sure that we don't err on the side
        # of identifying the onset too soon. Note that this guarantees we will
        # identify the onset too late!
        smoothed_audio_power = np.empty(audio_data.shape)
        audio_filter = np.ones((self.smoothing_filter_length,))
        for c in xrange(smoothed_audio_power.shape[1]):
            smoothed_audio_power[:,c] = sp.signal.lfilter(b=audio_filter,
                a=[1], x=audio_data[:,c]**2) / self.smoothing_filter_length
                
                        
        # now calculate the automatic threshhold
        if manual_threshhold is None:
            threshhold = self.autocalc_threshhold(smoothed_audio_power)
        elif manual_threshold < 1.0:
            # interpret this as a fraction of max
            threshhold = manual_threshold * smoothed_audio_power.max()
        else:
            # an exact threshhold
            threshhold = manual_threshhold
            
                    
        # find when smoothed waveform exceeds threshhold
        sound_bool_L = smoothed_audio_power[:,0] > threshhold
        sound_bool_R = smoothed_audio_power[:,1] > threshhold
        sound_bool = sound_bool_L | sound_bool_R
        
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
            assert (len(onsets) == len(offsets)) and (np.all(onsets < offsets))
                    
            # Remove sounds that violate min_duration requirement.
            too_short_sounds = ((offsets - onsets) < min_duration_samples)
            if np.any(too_short_sounds):
                print "Removing %d sounds that violate duration requirement" % \
                    len(mlab.find(too_short_sounds))
                
            onsets = onsets[np.logical_not(too_short_sounds)]
            offsets = offsets[np.logical_not(too_short_sounds)]
            
            # Warn when onsets occur very close together. This might occur if the 
            # sound power briefly drops below threshhold.
            if np.any(np.diff(onsets) < min_duration_samples):
                print "WARNING: %d onsets were suspiciously close together." % \
                    len(find(np.diff(onsets) < min_duration_samples))
            
            # Print the total number of sounds identified.
            print "Identified %d sounds with average duration %0.3fs" % \
                (len(onsets), (offsets-onsets).mean() * \
                (self.raw_data_loader.header.period / 30000.))
                
        
        # Now actually save and commit the onset times
        self.audio_onsets = onsets
        self.raw_data_loader.audio_onsets = self.audio_onsets        
        self.commit_audio_onsets()
        
        # Return unless debugging info is required.
        if not self.plot_debugging_figures:
            return
        
        
        # debugging figure
        # Plot all sound waveforms overlaid, to verify that they were caught
        # correctly.
        # This parameter determines how large the plotting window is.
        WINDOW_HALF_DURATION = 150 # samples
        
        # Initialize the figure and subplots.
        for s, side in enumerate(['left', 'right']):
            f = plt.figure()
            f.suptitle(side)
            ax = [f.add_subplot(2,2,n+1) for n in xrange(4)]            
            ax[0].set_title('Onset of sounds')
            ax[1].set_title('Offset of sounds')
            ax[2].set_title('Onset of smoothed')
            ax[3].set_title('Offset of smoothed')
                
            # Plot threshholds on the smoothed plots
            ax[2].semilogy([-WINDOW_HALF_DURATION, WINDOW_HALF_DURATION],\
                threshhold * np.ones((2,)), 'k:')
            ax[3].semilogy([-WINDOW_HALF_DURATION, WINDOW_HALF_DURATION],
                threshhold * np.ones((2,)), 'k:')
            for onset, offset in zip(onsets, offsets):
                # Plot a close-up of the onset
                ax[0].plot(\
                    np.arange(-WINDOW_HALF_DURATION, WINDOW_HALF_DURATION),
                    audio_data[onset-WINDOW_HALF_DURATION:
                    onset+WINDOW_HALF_DURATION, s])
                
                # Plot a close-up of the offset
                ax[1].plot(\
                    np.arange(-WINDOW_HALF_DURATION, WINDOW_HALF_DURATION),
                    audio_data[offset-WINDOW_HALF_DURATION:
                    offset+WINDOW_HALF_DURATION, s])
                
                # Now do the same but with the smoothed data
                ax[2].semilogy(\
                    np.arange(-WINDOW_HALF_DURATION, WINDOW_HALF_DURATION),
                    smoothed_audio_power[onset-WINDOW_HALF_DURATION:
                    onset+WINDOW_HALF_DURATION, s])
                
                ax[3].semilogy(\
                    np.arange(-WINDOW_HALF_DURATION, WINDOW_HALF_DURATION),
                    smoothed_audio_power[offset-WINDOW_HALF_DURATION:
                    offset+WINDOW_HALF_DURATION, s])

    
    def commit_audio_onsets(self):
        """Writes the newly detected audio onsets to disk.
        
        """
        # Note: even on a 32-bit system, this format allows >19hrs of indices.
        np.savetxt('audio_onsets', self.audio_onsets, '%i')
        self.raw_data_loader.audio_onsets = self.audio_onsets
        
    
    def autocalc_threshhold(self, filtered_audio):
        """Automatically calculates a reasonable threshhold to detect onsets.
        
        Theory
        ------
        First calculates the distribution of power in the audio signal
        across time. Presumably, for the great majority of the time,
        this power will be low. Occasionally, there will be infrequent
        bursts of audio power, which are the stimuli to be detected.
        
        There will be a range of power that almost never occurs, which
        is above baseline but below the least-powerful stimulus. This
        range will be a flat region in the cumulative distribution of
        power.
        
        This algorithm looks for that flat region and returns a
        threshhold right in the middle of it.
        """

        # first combine L and R to find one single threshhold
        # the reason for this is that chunks may occasionally contain 
        # audio on only
        # one channel, and then the threshhold will be way too low on the other
        # channel. more likely scenario is that threshhold can be 
        # approximately the
        # same on both channels. if this is not true, recode this, or use one of
        # the stupider auto-threshholding techniques
        
        
        # we will combine both channels
        # cumulative distribution of the data 
        cum_dist_amp = 20*np.log10(np.sort(filtered_audio.flatten()**2))
        
        # x-axis
        cum_dist_p = np.linspace(1, 0, len(cum_dist_amp))        

        # debugging figure
        # the figure shows threshhold on the x-axis, 
        # and the percentage of the data
        # ABOVE that threshhold on the y-axis
        # the threshhold is in dB = 20*log10(original_data)
        if self.plot_debugging_figures:
            f = plt.figure() 
            ax = f.add_subplot(111)
            ax.plot(cum_dist_amp, cum_dist_p);
            plt.show()

        # the algorithm takes some hard limits on how often the data can be above
        # threshhold
        MIN_TIME_ABOVE_THRESHHOLD = .001
        MAX_TIME_ABOVE_THRESHHOLD = .50
        search_regime = \
            (cum_dist_p > MIN_TIME_ABOVE_THRESHHOLD) & \
            (cum_dist_p < MAX_TIME_ABOVE_THRESHHOLD)
            
        idxs = np.arange(len(cum_dist_amp))            
        #idxs = (np.abs(np.diff(cum_dist_amp)) > 10*sys.float_info.epsilon)
        #temp, idxs = np.unique1d(np.diff(cum_dist_amp), return_index=True)
        cum_dist_amp2 = cum_dist_amp[idxs]
        search_regime2 = search_regime[idxs]
        cum_dist_p2 = cum_dist_p[idxs]
        
        cum_dist_amp2 = cum_dist_amp2[search_regime2]
        cum_dist_p2 = cum_dist_p2[search_regime2]
        
        
        # now find the slope of the distribution at each point
        # rise/run vs x-valuesf
        diff_cum_dist_amp2 = np.diff(cum_dist_p2) / np.diff(cum_dist_amp2)
       

        
        
        # and finds the threshhold which is LEAST sensitive to change in that
        # regime. should be a weighted average of the many good
        # threshholds, rather than the single best which is susceptible to weird
        # noise effects probably

        # define FOM. always negative because cumulative distribution slopes down.
        # the more negative it is, the LESS sensitive to perturbation
        
        
        figure_of_merit = 1. / diff_cum_dist_amp2
        
        # normalize and turn positive
        figure_of_merit = figure_of_merit / np.sum(figure_of_merit)
        
        
        # because the FOM was calculated with diff, it is one shorter than all
        # the other matrices. essentially each value is between two points.
        cum_dist_amp3 = cum_dist_amp2[:-1] + 0.5*np.diff(cum_dist_amp2)
        
        
        FOM_integral = np.cumsum(figure_of_merit*np.diff(cum_dist_amp2))
        FOM_integral = FOM_integral / FOM_integral[-1]
        
        
        
        # debugging figure. FOM should be maximally negative around the ideal
        # threshhold (within the search_regime2). 
        # it will probably also be quite noisy.
        if self.plot_debugging_figures:
            ax.plot(cum_dist_amp3, figure_of_merit)
            ax.plot(cum_dist_amp3, FOM_integral)
            plt.show()
        
        

        # average threshhold within the search regime WEIGHTED by the FOM
        th = cum_dist_amp3[np.argmin(np.abs(FOM_integral-0.5))]
        
        #th = np.sum(figure_of_merit * np.diff(cum_dist_amp2)) / \
        #    np.sum(figure_of_merit)
        #th = 
        
        if self.plot_debugging_figures:
            ax.plot([th,th], [0,1], 'k:')
            plt.show()
        
        return 10.**(th/20.)


class BehavingSyncer:
    """Syncs the audio onsets with other stimulus info in a DataSession.
    
    """
    def __init__(self):
        # I think I get all my info from containing DataSession?
        pass
    
    
    def sync(self, bcl, rdl, force_run=False):
        """Syncs the information in the neural (rdl) and behavioral (bcl) data.
        
        Line up the audio onsets in the two data sources.
        TODO: Consider adding behavioral information (eg stimulus type) to a
        common data store        
        
        
        Theory
        ------
        we have an accurate array of stimuli onsets from the behavior 
        computer and we have a less accurate array of detected stimuli
        onsets, produced by my code, operating on the captured speaker
        waveforms.
        
        There are two main synchronization issues with these files:
        1)  Recording began at different times, sometimes up to minutes
            apart. Thus, the arrays have different temporal offsets.
        2)  The recording computers have slightly different (and unknown) 
            clock frequencies. Generally the neural times need to be
            divided by ~0.99664 to match the behavior times.
        
        Also, the n_onsets may contain extra, spurious stimuli (such as the
        click associated with a reward), and may also be missing stimuli
        (if the event-detecting code missed a stimulus.) We will assume
        that b_onsets is "more accurate" and match the n_onsets to the
        b_onsets.
        
        First, the gross temporal offset between the files is estimated
        by smoothing the event timestamps and finding the maximum
        cross-correlations. Then, the event timestamps are matched up so
        that missing trials and spurious stimuli can be discarded. Finally,
        linear regression on the remaining trial times reveals the most
        accurate transformation from neural time to behavior time. This
        transformation is then stored to disk for other methods to use.
        
        
        Inputs
        ------
        bcl: bcontrol_loader
        rdl: raw_data_loader
        Generally these parameters are provided by the `sync` method of
        the DataSession that contains this syncer.
        
        
        Outputs (written to disk)
        -------------------------        
        CORR_NtoB : a linear polynomial used to fit the neural times to
        the behavior times. 
        
        CORR_BtoN : a linear polynomial providing the reverse transformation.
        
        
        Notes
        -----
        Output can be used like this:
        corr_n_times = polyval(CORR_NtoB, uncorr_n_times)
        corr_b_times = polyval(CORR_BtoN, uncorr_b_times)        
        """
        
        if os.path.exists('CORR_BtoN') and not force_run:
            self.CORR_BtoN = np.loadtxt('CORR_BtoN')
            self.CORR_NtoB = np.loadtxt('CORR_NtoB')
            print "Syncing information exists, great"
            return


        # gaussian filter event train
        # unnecessary to use original sampling rate because the other uncertainties
        # outweigh it. but you want to sample high enough that the gaussian from the
        # stimulus and the following reward click don't overlap (too much).
        SAMPLE_GAUSS = 500 # sample gaussians at this rate (Hz)


        # let's start with an approximately correct value for time dilation to make
        # things eaiser.
        ASSUMED_DILATION = .99664 # divide neural times by this value


        # use data from the beginning, when errors in ASSUMED_DILATION haven't had time
        # to add up yet. use as many trials as possible without taking forever to do the
        # cross-corr. also keep in mind that errors in dilation add up over this period.
        MAX_CORR_LENGTH = 500e3 # samples


        # if events are out of sync by more than this amount, they will not be discarded
        # instead of synced.
        MAXIMUM_ACCEPTABLE_TEMPORAL_ERROR = .010 # sec
        
        
        b_onsets = bcl.audio_onsets
        
        # find the last trial in behavior that is under this limit, ie, will
        # not exceed the maximum cross-correlation length once it is converted
        # to smoothed gaussian representation.
        try:
            LAST_TRIAL_B = np.nonzero(b_onsets - b_onsets[0] < \
                MAX_CORR_LENGTH / SAMPLE_GAUSS)[0][-1]
        except IndexError:
            LAST_TRIAL_B = len(b_onsets) - 1
        
        # Onsets in neural data
        n_onsets = rdl.audio_onsets / 30e3
        # in sec
        
        


        # find the last trial in neural data that is under this limit
        # Referenced to first neural trial
        # Note that first b trial and first n trial are actually pretty unlikely
        # to be the same. They just have to overlap with MAX_CORR_LENGTH
        
        
        try:
            LAST_TRIAL_N = mlab.find(n_onsets - n_onsets[0] > \
                b_onsets[LAST_TRIAL_B] - b_onsets[0])[0]
        except IndexError:
            LAST_TRIAL_N = len(n_onsets)-1
        
        
        # gaussian filter the event trains
        nn, xn = self.gauss_event_train( \
            (n_onsets[:LAST_TRIAL_N+1] / # The onset times 
            ASSUMED_DILATION * # Accounting for estimated dilation
            SAMPLE_GAUSS).round()) # Sampled at SAMPLE_GAUSS rate            
            
        nb, xb = self.gauss_event_train( \
            (b_onsets[:LAST_TRIAL_B+1] * # The onset times
            SAMPLE_GAUSS).round()) # Sample at SAMPLE_GAUSS rate

       
        # calculate cross-correlation and find maximum fit
        
        # Now, in theory zero-padding to a power of 2 and using
        # fft convolve should be optimal. But it doesn't work.
        # Seems awfully slow, even with zero-padding to power of 2:
        # next2 = np.ceil(max([np.log2(len(xn)), np.log2(len(xb))]))
        # xn2 = np.append(xn, np.zeros(2**next2 - len(xn)))
        # xb2 = np.append(xb, np.zeros(2**next2 - len(xb)))
        # C2 = sp.signal.fftconvolve(xn2, xb2, 'full')
        
        # Do a straight-up correlation
        same_length = max([len(xn), len(xb)])
        xn = np.append(xn, np.zeros(same_length - len(xn)))
        xb = np.append(xb, np.zeros(same_length - len(xb)))
        C = np.correlate(xn, xb, 'same')
        """        
        ODD LENGTH
        ----------
        In [21]: np.correlate([0,0,0,1,0], [0,0,1,0,0],'same')
        Out[21]: array([0, 0, 0, 1, 0])

        In [22]: np.correlate([0,0,1,0,0], [0,0,0,1,0],'same')
        Out[22]: array([0, 1, 0, 0, 0])
        
        So  argmax(C) < center : prepend zeros to first arg
        and argmax(C) > center : prepend zeros to second arg
        Number of zeros = abs((len(C)-1)/2 - argmax(C))
        
        
        EVEN LENGTH
        -----------
        In [24]: np.correlate([0,1,0,0], [0,0,1,0],'same')
        Out[24]: array([0, 1, 0, 0])

        In [25]: np.correlate([0,1,0,0], [0,1,0,0],'same')
        Out[25]: array([0, 0, 1, 0])

        In [26]: np.correlate([0,1,0,0], [1,0,0,0],'same')
        Out[26]: array([0, 0, 0, 1])     

        So  argmax(C) < center : prepend zeros to first arg
        and argmax(C) > center : prepend zeros to second arg
        Number of zeros = abs((len(C))/2 - argmax(C))
        
        
        GENERALIZED ODD AND EVEN
        ------------------------
        Delay = argmax(C) - len(C)/2    (Note: integer division)
        If delay < 0: prepend `delay` zeroes to first argument
        If delay > 0: prepend `delay` zeroes to second argument
        
        
        I cannot for the life of me figure out how to interpret the
        lags in the case where the arrays are not the same length.
                
        In [30]: np.correlate(np.array([0,0,1,0,0]), np.array([0,1,0]), 'same')
        Out[30]: array([0, 0, 1, 0, 0])

        In [31]: np.correlate(np.array([0,0,1,0,0]), np.array([0,1]), 'same')
        Out[31]: array([0, 0, 1, 0, 0])

        In [32]: np.correlate(np.array([0,0,1,0,0]), np.array([1,0]), 'same')
        Out[32]: array([0, 0, 0, 1, 0])

        In [33]: np.correlate(np.array([0,0,1,0,0]), np.array([1,0,0]), 'same')
        Out[33]: array([0, 0, 0, 1, 0])

        In [34]: np.correlate(np.array([0,0,1,0,0]), np.array([0,0,1,0,0]), 'same')
        Out[34]: array([0, 0, 1, 0, 0])

        In [35]: np.correlate(np.array([0,0,1,0,0]), np.array([0,0,1,0]), 'same')
        Out[35]: array([0, 0, 1, 0, 0])

        In [36]: np.correlate(np.array([0,0,1,0]), np.array([0,0,1]), 'same')
        Out[36]: array([0, 1, 0, 0])

        In [37]: np.correlate(np.array([0,0,1,0]), np.array([0,1]), 'same')
        Out[37]: array([0, 0, 1, 0])
        """
        
        # Calculate the best delay for the second argument (xb)
        best_offset_sec = np.float64((np.argmax(C) - len(C)/2)) / SAMPLE_GAUSS         
            

        """
        So the best line-up at this point is:
        corr_neural_time = ...
            neural_time / ASSUMED_DILATION - best_offset_sec -
            (nn[0]-nb[0]) / SAMPLE_GAUSS
        """
        # apply this correction to the neural data
        stim_onsets2 = \
            n_onsets / ASSUMED_DILATION \
            - best_offset_sec - (nn[0] - nb[0]) / SAMPLE_GAUSS
            
        
        # construct a matrix of synced times, one row for each neural stimulus
        synced_times = np.empty((len(stim_onsets2), 4))
        synced_times.fill(np.nan)

        # the third column is the original, uncorrected neural stimulus onsets
        synced_times[:,2] = n_onsets # uncorrected

        # and the second column is the corrected values
        synced_times[:,1] = stim_onsets2
       
        # For each behavioral time, find closest fit in neural onset time.
        # If no match is found, probably the audio onset was missed
        for nbt, bt in enumerate(b_onsets):
            best_idx_n = np.argmin(np.abs(bt - stim_onsets2))
            sync_err = bt - stim_onsets2[best_idx_n]
            if sync_err < MAXIMUM_ACCEPTABLE_TEMPORAL_ERROR:
                # A great match has been found
                synced_times[best_idx_n,0] = bt
        
        # Throw away all other neural onsets (probably spurious).
        good_idxs = np.logical_not(np.isnan(synced_times[:,0]))
                
        # now refit on real trials only.
        # fit behavior times synced_times(good_idxs,1) vs neural times
        CORR_NtoB = sp.polyfit(x=n_onsets[good_idxs], 
            y=synced_times[good_idxs, 0], deg=1)
        CORR_BtoN = sp.polyfit(y=n_onsets[good_idxs], 
            x=synced_times[good_idxs, 0], deg=1)
        
        
        # go back to original neural onsets
        # and apply the best fit to convert to behavior times        
        refit_b_times = sp.polyval(CORR_NtoB, n_onsets)
        refit_n_times = sp.polyval(CORR_BtoN, b_onsets)
        
        # Finally, verify that for each behavioral event, we can convert
        # this into a neural time that is within 10ms of a detected audio
        # event.
        
        # Let's say you know behavioral time of trial N is b_onsets[N]
        # and you want to know the time in the neural data.
        # The best guess is sp.polyval(CORR_BtoN, b_onsets[N])
        # And the index of the closest identified audio onset
        # in the neural data is map_b_to_n[N]. The actual audio onset is
        # n_onsets[map_b_to_n[N]], which should be within 10ms of
        # sp.polyval(CORR_BtoN, b_onsets[N])
        
        # Contrariwise:
        # Neural trial of interest: M
        # Neural time: n_onsets[M]
        # Estimated behavioral time: sp.polyval(CORR_NtoB, n_onsets[M])
        # Index of closest behavioral trial: map_n_to_b[M]
        # Actual time of closest behavioral trial: b_onsets[map_n_to_b[M]]
        # Error (<10ms): \
        #   abs(b_onsets[map_n_to_b[M]] - sp.polyval(CORR_NtoB, n_onsets[M]))
        map_b_to_n = np.array(\
            [np.argmin(np.abs(refit_b_times - bt)) for bt in b_onsets])
        map_n_to_b = np.array(\
            [np.argmin(np.abs(refit_n_times - nt)) for nt in n_onsets])

        n_errors = b_onsets[map_n_to_b] - sp.polyval(CORR_NtoB, n_onsets)
        b_errors = n_onsets[map_b_to_n] - sp.polyval(CORR_BtoN, b_onsets)
        
        map_b_to_n2 = np.ma.masked_array(map_b_to_n, \
            np.abs(b_errors) > MAXIMUM_ACCEPTABLE_TEMPORAL_ERROR)

        map_n_to_b2 = np.ma.masked_array(map_n_to_b, \
            np.abs(n_errors) > MAXIMUM_ACCEPTABLE_TEMPORAL_ERROR)
        
        
        # Print status information about the quality of the fit
        print "Using {0} behavioral trial onsets, identifed\n"\
        "{1} matching trials [min:{2},max:{3}] in the neural data".format\
        (len(b_onsets), len(np.where(map_b_to_n2.mask == 0)[0]),
        np.min(map_b_to_n2), np.max(map_b_to_n2))
        
        print "Using {0} neural trial onsets, identifed\n"\
        "{1} matching trials [min:{2},max:{3}] in the behavioral data".format\
        (len(n_onsets), len(np.where(map_n_to_b2.mask == 0)[0]),
        np.min(map_n_to_b2), np.max(map_n_to_b2))        
        
        
        # Store the results for later use
        self.CORR_NtoB = CORR_NtoB
        self.CORR_BtoN = CORR_BtoN
        self.map_b_to_n = map_b_to_n
        self.map_n_to_b = map_n_to_b
        self.map_b_to_n_masked = map_b_to_n2
        self.map_n_to_b_masked = map_n_to_b2
        
        self.commit()
    
    
    def commit(self):
        np.savetxt('CORR_NtoB', self.CORR_NtoB)
        np.savetxt('CORR_BtoN', self.CORR_BtoN)
        # I would like to save the index maps too but masked arrays don't
        # retain the mask when saved.
    
    
    def gauss_event_train( self, timestamps, filter_std=10, 
        filter_truncation_width=None,
        n_min=None, n_max=None): 
        """Returns a filtered time series representation.
        
        BEHAVIOR
        --------
        For a specified list of event times, returns a filtered time series
        representation as a tuple: (n_op, x_op).  
        
        The time points are given in n_op (in samples) and the values
        are given in x_op. A gaussian will be added to x_op centered at each
        timestamp.
        
        There is no support for F_SAMP. You must specify the event times
        and all other parameters in samples.
        
        
        INPUTS
        ------
        timestamps: ndarray of time values (IN SAMPLES!!) where events occurred
        
        
        OPTIONAL INPUTS
        ---------------
        filter_std: Standard deviation (width) of the Gaussian, in samples.
        The default value is ten.
               
        filter_truncation_width: probably don't mess with this. 
        The gaussians are finite in extent. default is 5 standard deviations.
        
        n_min : force n_op.min() to be at most this, will not truncate
        n_max : force n_op.max() to be at least this, will not truncate
        
        
        OUTPUTS
        -------
        n_op: the samples of each point. n_op ranges from the minimum
        timestamp to the maximum timestamp, eg:
        arange(np.min(timestamps), np.max(timestamps)+1)
        Note that this means that the first and last Gaussians will
        be half-truncated!
        
        x_op: the value of the smoothed function at each sample.        
        """
        
        # Finalize the default values of parameters    
        if filter_truncation_width is None: 
            filter_truncation_width = 5*filter_std
        
        # Convert timestamps to integers for use in indexing
        timestamps = np.int32(timestamps)
        
        # Determine the range of the output
        start_sample = np.min(timestamps) - filter_truncation_width
        stop_sample = np.max(timestamps) + filter_truncation_width
        if n_min is not None and n_min < start_sample:
            start_sample = n_min
        if n_max is not None and n_max > stop_sample:
            stop_sample = n_max
               
        # generate normalized gaussian on n_gauss and x_gauss
        n_gauss = np.arange(-filter_truncation_width,
            filter_truncation_width + 1)
        x_gauss = np.exp( -(np.float64(n_gauss) ** 2) / (2 * filter_std**2) )
        x_gauss = x_gauss / np.sum(x_gauss)
        
        # initialize return variables        
        n_op = np.arange(start_sample, stop_sample + 1)
        x_op = np.zeros(n_op.shape) # value, float64
        
        # for each timestamp, add a gaussian to x_op
        for timestamp in timestamps:
            # calculate where we'd ideally like to start and stop
            # adding the gaussian
            x_op[timestamp - start_sample + n_gauss] += x_gauss
        
        return (n_op, x_op)
