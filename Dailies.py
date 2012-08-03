import numpy as np
from myutils import printnow
import RecordingSession
import myutils
import RecordingSessionMaker as rswrap
import matplotlib.pyplot as plt
import os
import datetime
import glob

def gettime(filename):
    return datetime.datetime.fromtimestamp(os.path.getmtime(filename))

def before(l, target, N=None, include_zero=True):
    """Returns N sorted indexes of entries in list that occur before target."""
    la = np.asarray(l)
    idxs = np.argsort(target - la)
    if include_zero:
        idxs2 = np.array(filter(lambda li: la[li] <= target, idxs))
    else:
        idxs2 = np.array(filter(lambda li: la[li] < target, idxs))
    
    if N is None:
        N = len(idxs2)
    
    return idxs2[:N]

def run_tones(rs=None, 
    output_dir='/media/dendrite',
    all_channels_file='../DO1_ALL_CHANNELS', 
    channel_groups_file='../DO1_CHANNEL_GROUPS', 
    analog_channels_file='../ANALOG_7_8',
    ns5_filename=None,
    remove_from_TC=None,
    soft_time_limits=(-1.0, 1.0),
    hard_time_limits=(-.04, .15),
    do_add_bcontrol=True,
    bcontrol_folder='/media/hippocampus/TRWINRIG_DATA/Data/SPEAKERCAL/',
    bcontrol_files=None,
    n_tones=None,
    TR_NDAQ_offset_sec=420,
    start_offset=0,
    stop_offset=0,
    do_timestamps=True,
    plot_spectrograms=True,
    break_at_spectrograms=False,
    force_put_neural_data=False,
    do_avg_plots=True,
    do_extract_spikes=True,
    detection_kwargs=None,
    CAR=True,
    save_to_klusters=True,
    do_MUA_grand_plot=True,
    group_multiplier=100,
    psth_time_limits=(None,None),
    do_tuning_curve=True,
    **kwargs):
    """Daily run script for tones (tuning curve)
    
    rs : RS if it exists. If it doesn't, provide the following:
        output_dir, all_channels_file, channel_groups_file, 
        analog_channels_file, ns5_filename, remove_from_TC, soft_time_limits,
        hard_time_limits
    
    do_add_bcontrol: if True, will find bcontrol files, extract information,
        write "tones" and "attens" to directory. (If those files already exist,
        then this block is skipped, so delete them if you want to force.)

        You can specify explicitly, or else it will search.
        
        bcontrol_files : a list of bcontrol files that you've selected.
        
        bcontrol_folder : If bcontrol_files is None, then will look here
            for them. Will try to guess from ns5 time which are appropriate.
            It will keep grabbing files until it find at least `n_tones`
            tones. If n_tones is None, uses the number of timestamps.

        In either case, the mat files are copied into the directory, and then
        the `tones` and `attens` files are written. Those files are used
        for all subsequent analyses.
    
    plot_spectrograms: if True, will plot spectrograms of the audio stimulus
        for every 200 tones, in order to check that the tones and attens 
        are correctly lined up.
        
        break_at_spectrograms : if True, errors immediately after plotting
        spectrograms, for debugging
    
    do_tuning_curve : if True, plots tuning curve
    
    Other parameters should be same as other Dailies.
    """
    if len(kwargs) > 0:
        print "unexpected kwargs"
        print kwargs
    
    # Make session
    if rs is None:
        printnow("creating recording session %s" % ns5_filename)
        rsm = rswrap.RecordingSessionMaker(
            data_analysis_dir=output_dir,
            all_channels_file=all_channels_file,
            channel_groups_file=channel_groups_file,
            analog_channels_file=analog_channels_file)

        if remove_from_TC is None:
            remove_from_TC = []

        rs = rsm.make_session(
            ns5_filename=ns5_filename,
            remove_from_TC=remove_from_TC)
    
        rs.write_time_limits(soft_time_limits, hard_time_limits)
    rs.group_multiplier = group_multiplier
    printnow("RS %s" % rs.full_path)

    # add timestamps
    if do_timestamps:
        printnow("adding timestamps")
        # write timestamps to directory
        # have to force, otherwise will sub-time it twice
        times, numbers = rswrap.add_timestamps_to_session(rs, verbose=True, 
            force=True, meth='digital_trial_number')
        
        # sub time
        subtimes = np.rint(np.linspace(3300, 1194210, 200)).astype(np.int)
        alltimes = np.concatenate([time + subtimes for time in times])
        rs.add_timestamps(alltimes)
    
    # add bcontrol
    tone_filename = os.path.join(rs.full_path, 'tones')
    atten_filename = os.path.join(rs.full_path, 'attens')
    if do_add_bcontrol and (not os.path.exists(tone_filename) or not \
        os.path.exists(atten_filename)):
        printnow('adding bcontrol')
        
        # First find out how many tones there probably are
        if n_tones is None:
            n_tones = len(rs.read_timestamps())
        
        if bcontrol_files is None:
            # Guess by ns5 filetime
            ns5_stoptime = gettime(rs.get_ns5_filename())
            ns5_startime = ns5_stoptime - datetime.timedelta(seconds=
                rs.get_ns5_loader().header.n_samples / rs.get_sampling_rate())
            ns5_stoptime += datetime.timedelta(seconds=TR_NDAQ_offset_sec)
            ns5_startime += datetime.timedelta(seconds=TR_NDAQ_offset_sec)
            mintime = ns5_startime + datetime.timedelta(seconds=start_offset)
            maxtime = ns5_stoptime + datetime.timedelta(seconds=stop_offset)
            
            # Find the bcontrol files that were saved during the recording
            # And sort by time
            allfiles = np.asarray(glob.glob(os.path.join(
                bcontrol_folder, 'speakercal*.mat')))
            bcontrol_filetimes = np.asarray(map(gettime, allfiles))
            sidxs = np.argsort(bcontrol_filetimes)
            bcontrol_filetimes = bcontrol_filetimes[sidxs]
            allfiles = allfiles[sidxs]
            
            # Choose the files within the window
            check_idxs = np.where(
                (bcontrol_filetimes > mintime) & 
                (bcontrol_filetimes < maxtime))[0]
            
            # Iterate through the found files until a sufficient number
            # of tones have been found
            n_found_tones = 0
            found_files = []
            for check_idx in check_idxs:
                # Load file
                filename = allfiles[check_idx]
                tl = myutils.ToneLoader(filename)
                
                # Skip if WN
                if not np.all(tl.tones == 0):
                    found_files.append(filename)
                    n_found_tones += len(tl.tones)
                
                # Break if enough found
                if n_found_tones >= n_tones:
                    break

            # Output debugging info
            print "I found %d tones in %d files" % (
                n_found_tones, len(found_files))
            if n_found_tones < n_tones:
                print "insufficient tones found ... try increasing start delta"
            
            # More debugging info about first file
            print "Using general offset of " + str(TR_NDAQ_offset_sec) + " ...."
            idx1 = np.where(allfiles == found_files[0])[0]
            offsets = bcontrol_filetimes[idx1-1:idx1+2] - ns5_startime
            poffsets1 = [offset.seconds if offset > datetime.timedelta(0) 
                else -(-offset).seconds for offset in offsets]
            print "First file (prev,curr,next) offsets from start: %d %d %d" % \
                (poffsets1[0], poffsets1[1], poffsets1[2])
            
            # And last file
            idx1 = np.where(allfiles == found_files[-1])[0]
            offsets = bcontrol_filetimes[idx1-1:idx1+2] - ns5_stoptime
            poffsets2 = [offset.seconds if offset > datetime.timedelta(0) 
                else -(-offset).seconds for offset in offsets]
            print "Last file (prev,curr,next) offsets from stop: %d %d %d" % \
                (poffsets2[0], poffsets2[1], poffsets2[2])

            # Now put in forward order
            bcontrol_files = np.asarray(found_files)
            
            # Debugging output
            print "Like these results? Here's how to replicate:"
            print "<speakercal_files>"
            for bcf in bcontrol_files:
                print os.path.split(bcf)[1]
            print "</speakercal_files>"
            print "clock_offset='%d' start_offset='%d %d %d' stop_offset='%d %d %d'" % (
                TR_NDAQ_offset_sec, 
                poffsets1[0], start_offset, poffsets1[1], 
                poffsets2[1], stop_offset, poffsets2[2])
        
        # Add to RS
        if bcontrol_files is not None:
            for file in bcontrol_files:
                rs.add_file(file)
    
        # Now that we've settled on a canonical bcontrol file ordering,
        # dump tones and attens
        tls = [myutils.ToneLoader(file) for file in bcontrol_files]
        tones = np.concatenate([tl.tones for tl in tls])
        attens = np.concatenate([tl.attens for tl in tls])  
        np.savetxt(tone_filename, tones)
        np.savetxt(atten_filename, attens, fmt='%d')
        

    if plot_spectrograms:
        tones = np.loadtxt(tone_filename)
        attens = np.loadtxt(atten_filename, dtype=np.int)
        
        # verify timestamps
        timestamps = rs.read_timestamps()
        if len(timestamps) < len(tones):
            print "warning not enough timestamps, discarding tones: " + \
                "%d timestamps but %d tones" % (
                len(timestamps), len(tones))
            tones = tones[:len(timestamps)]
            attens = attens[:len(timestamps)]
        elif len(timestamps) > len(tones):
            print "warning too many timestamps, provide more tones: " + \
                "%d timestamps but %d tones" % (
                len(timestamps), len(tones))
        
        # check spectrograms
        # plot debugging spectrograms of audio
        l = rs.get_ns5_loader()
        raw = l.get_chunk_by_channel()
        ain135 = raw[135]
        ain136 = raw[136]
        
        # Spectrogrammer object
        sg = myutils.Spectrogrammer(NFFT=1024, Fs=30e3, max_freq=30e3, 
            min_freq=0, noverlap=512, downsample_ratio=1)
        
        # Fake toneloader to calculate aliased tones
        tl = myutils.ToneLoader()
        tl.tones = tones
        aliased_tones = tl.aliased_tones()
        
        for n in range(0, len(tones) - 5, 200):
            ts = timestamps[n]
            known_tones = aliased_tones[n:n+5]
            slc1 = ain135[ts:ts+30e3]
            slc2 = ain136[ts:ts+30e3]
            
            # Transform and plot
            Pxx, freqs, t = sg.transform(np.mean([slc1, slc2], axis=0))
            myutils.my_imshow(Pxx, t, freqs)
            plt.axis('auto')
            
            # Title with known tones
            plt.title('tl%d %0.1f %0.1f %0.1f %0.1f %0.1f' % (
                n, known_tones[0], known_tones[1], known_tones[2], 
                known_tones[3], known_tones[4]))
            
            # Save to RS
            plt.savefig(os.path.join(rs.full_path, 'tones_%d.png' % n))
            plt.close()
        
        if break_at_spectrograms:
            1/0

    # put in neural db (does nothing if exists unless forced)
    printnow('putting neural data')
    rs.put_neural_data_into_db(verbose=True, force=force_put_neural_data)

    # plot averages
    if do_avg_plots:
        printnow("avg plots")
        rswrap.plot_avg_lfp(rs, savefig=True)
        rswrap.plot_avg_audio(rs, savefig=True)

    # spike extract
    if do_extract_spikes:
        printnow('extracting spikes')
        rs.generate_spike_block(CAR=CAR, smooth_spikes=False, verbose=True)
        rs.run_spikesorter(save_to_db=True, save_to_klusters=save_to_klusters,
            detection_kwargs=detection_kwargs)
        rs.spiketime_dump()

    # plot MUA stuff
    if do_MUA_grand_plot:
        printnow('mua grand psths')        
        rswrap.plot_all_spike_psths(rs, savefig=True)
        
    
    # make a tuning curve
    if do_tuning_curve:
        # extract tones and attens from each
        tones = np.loadtxt(tone_filename)
        attens = np.loadtxt(atten_filename, dtype=np.int)
        if len(timestamps) < len(tones):
            print "warning not enough timestamps, discarding tones: " + \
                "%d timestamps but %d tones" % (
                len(timestamps), len(tones))
            tones = tones[:len(timestamps)]
            attens = attens[:len(timestamps)]
        elif len(timestamps) > len(tones):
            print "warning too many timestamps, provide more tones: " + \
                "%d timestamps but %d tones" % (
                len(timestamps), len(tones))        
        
        
        # parameters for tuning curve
        tc_freqs = 10 ** np.linspace(np.log10(5e3), np.log10(50e3), 15)
        tc_attens = np.unique(attens)

        # Determine which bin each trial belongs to
        tone_freq_bin = np.searchsorted(tc_freqs, tones, side='right') - 1
        tone_atten_bin = np.searchsorted(tc_attens, attens, side='right') - 1
        
        # spike count for each trial
        group = 5
        spike_time_file = os.path.join(rs.last_klusters_dir(),
            '%s.res.%d' % (rs.session_name, group))
        spike_times = np.loadtxt(spike_time_file, dtype=np.int)
        timestamps = rs.read_timestamps()
        spike_counts = count_within_window(timestamps, spike_times,
            .005*30e3, .030*30e3)
        
        # reshape into tuning curve
        tc_mean = np.zeros((len(tc_attens), len(tc_freqs) - 1))
        tc_std = np.zeros((len(tc_attens), len(tc_freqs) - 1))        
        tc_median = np.zeros((len(tc_attens), len(tc_freqs) - 1))        
        for n, tc_freq in enumerate(tc_freqs[:-1]):
            for m, tc_atten in enumerate(tc_attens):
                # Which tones go into this bin
                tone_idxs = np.where(
                    (tone_freq_bin == n) & (tone_atten_bin == m))[0]
                if len(tone_idxs) == 0:
                    print "none in this bin %f %d" % (tc_freq, tc_atten)
                    continue        
                
                tc_mean[m, n] = np.mean(spike_counts[tone_idxs])
                tc_median[m, n] = np.median(spike_counts[tone_idxs])
                tc_std[m, n] = np.std(spike_counts[tone_idxs])
        
        # plot
        np.savez('data', tc_mean=tc_mean, tc_freqs=tc_freqs, tc_attens=tc_attens)
        myutils.my_imshow(tc_mean, tc_freqs, tc_attens, cmap=plt.cm.Purples)
        plt.axis('tight')
        plt.colorbar()
        myutils.my_imshow(tc_median, tc_freqs, tc_attens, cmap=plt.cm.Purples)
        plt.colorbar()
        myutils.my_imshow(tc_std, tc_freqs, tc_attens, cmap=plt.cm.gray)
        plt.colorbar()
        plt.show()
    
    return rs

def count_within_window(timestamps, spike_times, winstart=0, winstop=1):
    winstart = int(winstart)
    winstop = int(winstop)
    
    count_l = []
    for ts in timestamps:
        count_l.append(len(spike_times[
            (spike_times > ts + winstart) & (spike_times < ts + winstop)]))
    
    return np.asarray(count_l)
    

def run_tonetask(rs=None, 
    output_dir='/media/dendrite',
    all_channels_file='../DO1_ALL_CHANNELS', 
    channel_groups_file='../DO1_CHANNEL_GROUPS', 
    analog_channels_file='../ANALOG_7_8',
    ns5_filename=None,
    remove_from_TC=None,
    soft_time_limits=(-2000., 2000.),
    hard_time_limits=(-.04, .15),
    do_timestamps=False,
    force_put_neural_data=False,
    do_avg_plots=True,
    do_extract_spikes=True,
    CAR=True,
    save_to_klusters=True,
    do_MUA_grand_plot=True,
    do_run_klustakwik=False,
    group_multiplier=100,
    **kwargs
    ):
    if len(kwargs) > 0:
        print "unexpected kwargs"
        print kwargs
    
    # Make session
    if rs is None:
        printnow("creating recording session %s" % ns5_filename)
        rsm = rswrap.RecordingSessionMaker(
            data_analysis_dir=output_dir,
            all_channels_file=all_channels_file,
            channel_groups_file=channel_groups_file,
            analog_channels_file=analog_channels_file)

        if remove_from_TC is None:
            remove_from_TC = []

        rs = rsm.make_session(
            ns5_filename=ns5_filename,
            remove_from_TC=remove_from_TC)
    
        rs.write_time_limits(soft_time_limits, hard_time_limits)
    printnow("RS %s" % rs.full_path)

    # add timestamps
    if do_timestamps:
        1/0
        printnow("adding timestamps")
        # write timestamps to directory
        times, numbers = rswrap.add_timestamps_to_session(rs, verbose=True, 
            meth='audio_onset')
    
    # put in neural db (does nothing if exists unless forced)
    printnow('putting neural data')
    rs.put_neural_data_into_db(verbose=True, force=force_put_neural_data)
    
    # plot averages
    if do_avg_plots:
        printnow("avg plots")
        rswrap.plot_avg_lfp(rs, savefig=True)
        rswrap.plot_avg_audio(rs, savefig=True)

    # spike extract
    if do_extract_spikes:
        printnow('extracting spikes')
        rs.generate_spike_block(CAR=CAR, smooth_spikes=False, verbose=True)
        rs.run_spikesorter(save_to_db=True, save_to_klusters=save_to_klusters)
        rs.spiketime_dump()

    # plot MUA stuff
    if do_MUA_grand_plot:
        printnow('mua grand psths')
        rswrap.plot_all_spike_psths(rs, savefig=True)

    # spike sort
    if do_run_klustakwik:
        printnow('running klustakwik')
        rs.group_multiplier = group_multiplier
        rs.run_klustakwik()
        rs.prep_for_klusters(verbose=True)

    return rs


def run_wn(rs=None, 
    output_dir='/media/dendrite',
    all_channels_file='../DO1_ALL_CHANNELS', 
    channel_groups_file='../DO1_CHANNEL_GROUPS', 
    analog_channels_file='../ANALOG_7_8',
    ns5_filename=None,
    remove_from_TC=None,
    soft_time_limits=(-2., 2.),
    hard_time_limits=(-.04, .15),
    do_timestamps=True,
    force_put_neural_data=False,
    do_avg_plots=True,
    do_extract_spikes=True,
    CAR=True,
    save_to_klusters=True,
    do_MUA_grand_plot=True,
    do_run_klustakwik=True,
    group_multiplier=100,
    **kwargs
    ):
    if len(kwargs) > 0:
        print "unexpected kwargs"
        print kwargs
    
    # Make session
    if rs is None:
        printnow("creating recording session %s" % ns5_filename)
        rsm = rswrap.RecordingSessionMaker(
            data_analysis_dir=output_dir,
            all_channels_file=all_channels_file,
            channel_groups_file=channel_groups_file,
            analog_channels_file=analog_channels_file)

        if remove_from_TC is None:
            remove_from_TC = []

        rs = rsm.make_session(
            ns5_filename=ns5_filename,
            remove_from_TC=remove_from_TC)
    
        rs.write_time_limits(soft_time_limits, hard_time_limits)
    printnow("RS %s" % rs.full_path)

    # add timestamps
    if do_timestamps:
        printnow("adding timestamps")
        # write timestamps to directory
        times, numbers = rswrap.add_timestamps_to_session(rs, verbose=True, 
            meth='audio_onset')
    
    # put in neural db (does nothing if exists unless forced)
    printnow('putting neural data')
    rs.put_neural_data_into_db(verbose=True, force=force_put_neural_data)
    
    # plot averages
    if do_avg_plots:
        printnow("avg plots")
        rswrap.plot_avg_lfp(rs, savefig=True)
        rswrap.plot_avg_audio(rs, savefig=True)

    # spike extract
    if do_extract_spikes:
        printnow('extracting spikes')
        rs.generate_spike_block(CAR=CAR, smooth_spikes=False, verbose=True)
        rs.run_spikesorter(save_to_db=True, save_to_klusters=save_to_klusters)
        rs.spiketime_dump()

    # plot MUA stuff
    if do_MUA_grand_plot:
        printnow('mua grand psths')
        rswrap.plot_all_spike_psths(rs, savefig=True)

    # spike sort
    if do_run_klustakwik:
        printnow('running klustakwik')
        rs.group_multiplier = group_multiplier
        rs.run_klustakwik()
        rs.prep_for_klusters(verbose=True)

    return rs

    
def run_behaving(rs=None, 
    output_dir='/media/dendrite',
    all_channels_file='../DO1_ALL_CHANNELS', 
    channel_groups_file='../DO1_CHANNEL_GROUPS', 
    analog_channels_file='../ANALOG_7_8',
    ns5_filename=None,
    remove_from_TC=None,
    soft_time_limits=(-2., 2.),
    hard_time_limits=(-.04, .15),
    do_add_bcontrol=True,
    bcontrol_folder='/media/hippocampus/TRWINRIG_DATA/Data/chris/CR20B',
    do_timestamps=True,
    do_add_btrial_numbers=True,
    force_put_neural_data=False,
    do_avg_plots=True,
    do_extract_spikes=True,
    CAR=True,
    save_to_klusters=True,
    do_MUA_grand_plot=True,
    do_MUA_plot_by_stimulus=True,
    do_run_klustakwik=True,
    do_SUA_plot_by_stimulus=True,
    group_multiplier=100,
    psth_time_limits=(None,None),
    lfp_time_limits='hard',
    **kwargs
    ):
    if len(kwargs) > 0:
        print "unexpected kwargs"
        print kwargs
    
    # Make session
    if rs is None:
        printnow("creating recording session %s" % ns5_filename)
        rsm = rswrap.RecordingSessionMaker(
            data_analysis_dir=output_dir,
            all_channels_file=all_channels_file,
            channel_groups_file=channel_groups_file,
            analog_channels_file=analog_channels_file)

        if remove_from_TC is None:
            remove_from_TC = []

        rs = rsm.make_session(
            ns5_filename=ns5_filename,
            remove_from_TC=remove_from_TC)
    
        rs.write_time_limits(soft_time_limits, hard_time_limits)
    rs.group_multiplier = group_multiplier
    printnow("RS %s" % rs.full_path)

    # add bcontrol
    if do_add_bcontrol:
        printnow('adding bcontrol')
        rswrap.add_bcontrol_data_to_session(
            bcontrol_folder, rs, verbose=True)

    # add timestamps
    if do_timestamps:
        printnow("adding timestamps")
        # write timestamps to directory
        times, numbers = rswrap.add_timestamps_to_session(rs, verbose=True, 
            meth='digital_trial_number')

    # put in neural db (does nothing if exists unless forced)
    printnow('putting neural data')
    rs.put_neural_data_into_db(verbose=True, force=force_put_neural_data)

    # adding btrial numbers
    if do_add_btrial_numbers:
        printnow('adding btrial numbers')
        if len(numbers) == 0:
            print "warning: no numbers?"
            numbers = None # will auto-determine if all is well
        rswrap.add_behavioral_trial_numbers2(rs, known_trial_numbers=numbers, 
            trial_number_channel=16, verbose=True)
    
    # plot averages
    if do_avg_plots:
        printnow("avg plots")
        if lfp_time_limits == 'hard':
            lfp_time_limits = rs.read_time_limits[1]
        
        rswrap.plot_avg_lfp(rs, savefig=True, 
            t_start=lfp_time_limits[0], t_stop=lfp_time_limits[1])
        rswrap.plot_avg_audio(rs, savefig=True)

    # spike extract
    if do_extract_spikes:
        printnow('extracting spikes')
        rs.generate_spike_block(CAR=CAR, smooth_spikes=False, verbose=True)
        rs.run_spikesorter(save_to_db=True, save_to_klusters=save_to_klusters)
        rs.spiketime_dump()

    # plot MUA stuff
    if do_MUA_grand_plot:
        printnow('mua grand psths')        
        rswrap.plot_all_spike_psths(rs, savefig=True)

    # plot MUA by stim
    if do_MUA_plot_by_stimulus:
        printnow("mua by stimulus")
        t_start, t_stop = psth_time_limits
        rswrap.plot_MUA_by_stim(rs, savefig=True, 
            t_start=t_start, t_stop=t_stop)

    # spike sort
    if do_run_klustakwik:
        printnow('running klustakwik')
        rs.group_multiplier = group_multiplier
        rs.run_klustakwik()
        rs.prep_for_klusters(verbose=True)
    
    # SUA by stim
    if do_SUA_plot_by_stimulus:
        printnow("SUA plot by stimulus")
        t_start, t_stop = psth_time_limits
        rswrap.plot_all_spike_psths_by_stim(rs, savefig=True, 
            t_start=t_start, t_stop=t_stop,
            skipNoScore=False, override_path=rs.last_klusters_dir())

    return rs