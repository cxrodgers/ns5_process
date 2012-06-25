import numpy as np
from myutils import printnow
import RecordingSession
import myutils
import RecordingSessionMaker as rswrap
import matplotlib.pyplot as plt
import os

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
    bcontrol_folder='/media/hippocampus/TRWINRIG_DATA/Data/chris/CR20B',
    bcontrol_files=None,
    do_timestamps=True,
    break_at_spectrograms=False,
    force_put_neural_data=False,
    do_avg_plots=True,
    do_extract_spikes=True,
    detection_kwargs=None,
    CAR=True,
    save_to_klusters=True,
    do_MUA_grand_plot=True,
    do_run_klustakwik=True,
    group_multiplier=100,
    psth_time_limits=(None,None),
    do_tuning_curve=True,
    **kwargs):
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
    if do_add_bcontrol:
        printnow('adding bcontrol')
        if bcontrol_files is None:
            # sort the bcontrol files by datetime
            TR_NDAQ_offset = datetime.timedelta(minutes=7)

            # Find the bcontrol files that were saved during the recording
            allfiles = np.asarray(glob.glob(os.path.join(
                bcontrol_folder, 'speakercal*.mat')))
            bcontrol_filetimes = map(gettime, allfiles)
            idxs = before(bcontrol_filetimes, ns5time + TR_NDAQ_offset, N=None)

            # Now put in forward order, and discard the most recent one which was
            # not played during the recording
            bcontrol_files = allfiles[idxs][:0:-1]
        

        if bcontrol_files is not None:
            for file in bcontrol_files:
                rs.add_file(file)
        

        # Now load data from them
        tls = [myutils.ToneLoader(file) for file in usefiles]
        tones = np.concatenate([tl.tones for tl in tls])
        attens = np.concatenate([tl.attens for tl in tls])  
        
        
        # extract tones and attens from each
        #tls = [myutils.ToneLoader(file) for file in bcontrol_files[:-len(times)-1:-1]]
        tls = [myutils.ToneLoader(file) for file in bcontrol_files]
        tones = np.concatenate([tl.tones for tl in tls])
        attens = np.concatenate([tl.attens for tl in tls])
        
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
        sg = myutils.Spectrogrammer(NFFT=1024, Fs=30e3, max_freq=30e3, 
            min_freq=0, noverlap=512, downsample_ratio=1)
        for n, tl in enumerate(tls):
            ts = timestamps[200*n]
            known_tones = tl.aliased_tones()[:5]
            slc1 = ain135[ts:ts+30e3]
            slc2 = ain136[ts:ts+30e3]
            Pxx, freqs, t = sg.transform(np.mean([slc1, slc2], axis=0))
            myutils.my_imshow(Pxx, t, freqs)
            plt.axis('auto')
                
            plt.title('tl%d %0.1f %0.1f %0.1f %0.1f %0.1f' % (
                n, known_tones[0], known_tones[1], known_tones[2], 
                known_tones[3], known_tones[4]))
            
            plt.savefig(os.path.join(rs.full_path, 'tone_tl_%d.png' % n))
            plt.close()
        
        if break_at_spectrograms:
            1/0
        
        # put metadata into db? or save tones/attens to directory?

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
        tls = [myutils.ToneLoader(file) for file in bcontrol_files]
        tones = np.concatenate([tl.tones for tl in tls])
        attens = np.concatenate([tl.attens for tl in tls])        
        
        # parameters for tuning curve
        tc_freqs = 10 ** np.linspace(np.log10(5e3), np.log10(50e3), 15)
        tc_attens = np.unique(attens)

        # Determine which bin each trial belongs to
        tone_freq_bin = np.searchsorted(tc_freqs, tones, side='right') - 1
        tone_atten_bin = np.searchsorted(tc_attens, attens, side='right') - 1
        
        # spike count for each trial
        group = 7
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