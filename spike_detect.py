    OE_db = glob.glob(os.path.join(data_dir, '*.db'))[0]
    metadata = OE.open_db("sqlite:///" + OE_db)
    session = OE.Session()
    query = session.query(OE.RecordingPoint)

    # Find groups
    group_list = list(np.unique([rp.group for rp in query.all()]))
    if None in group_list: group_list.remove(None)


    # Sort each group
    for group in group_list:
        print "Sorting tetrode %d" % group
        
        # Get RecordingPoint on this group
        rp_list = query.filter((OE.RecordingPoint.group == group) and \
            (OE.RecordingPoint.block.name == 'CAR Tetrode Data')).all()
        
        # Create spikesorter for this group
        spikesorter = OE.SpikeSorter(mode='from_filtered_signal', 
            session=session, recordingPointList=rp_list)
        
        # Detect spikes
        spikesorter.computeDetection(OE.detection.MedianThreshold, 
            sign='-', median_thresh=5., left_sweep=.001, right_sweep=.002)

        # Dump results to OE db
        spikesorter.save_to_db()