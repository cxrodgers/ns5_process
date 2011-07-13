"""Helper functions and variables for all data in a recording session.

This allows you to create a well-formed directory for ns5_process and fill
it with data.

Try to keep logic relating to any specific experiment out of this module.

RecordingSession spec:
* A directory
* File containing neural channels to put into database
  16 17 18 20 22 24 26 28
* File containing channel groupings
  16 17 18 20
  22 24 26 28
* File containing analog channels to put into database (if any)
  7 8
* TIMESTAMPS with times in samples to extract
* Time limits filename with soft limits on first line, then hard
"""

import shutil
import glob
import os.path
import ns5
import numpy as np
import TrialSlicer
import OpenElectrophy as OE
import matplotlib.pyplot as plt
ALL_CHANNELS_FILENAME = 'NEURAL_CHANNELS_TO_GET'
GROUPED_CHANNELS_FILENAME = 'NEURAL_CHANNEL_GROUPINGS'
ANALOG_CHANNELS_FILENAME = 'ANALOG_CHANNELS'
TIMESTAMPS_FILENAME = 'TIMESTAMPS'
TIME_LIMITS_FILENAME = 'TIME_LIMITS'
FULL_RANGE_UV = 8192. # maximum input range of signal

def write_channel_numbers(filename, list_of_lists):
    """Writes a list of lists of channel numbers to a file.
    
    Each list in list_of_lists in written to a line of the file.
    Each line contains channel numbers with a single space following each.
    
    Actually I started using this for all kinds of writes, it works
    like np.save and np.load except there is no requirement that each
    entry be the same length.
    """
    fi = file(filename, 'w')
    for chlist in list_of_lists:
        for ch in chlist: 
            fi.write(str(ch) + ' ')
        fi.write('\n')
    fi.close()

def read_channel_numbers(filename, dtype=np.int):
    """ Load TETRODE_CHANNELS control file
    This is funny-looking because not all tetrodes have same number
    of channels!
    """
    f = file(filename)
    x = f.readlines()
    f.close()
    return [[dtype(c) for c in r] for r in [str.split(rr) for rr in x]]

class RecordingSession:
    """Object linked to a directory containing data for processing.
    
    Provides methods to read and write data from that directory.
    
    Some methods look for stereotype filenames which are defined by
    globals above. Others find a target by globbing. The latter provides
    getter methods and return None if the file doesn't exist. Perhaps
    the former should provide getter methods that always work, even 
    if file doesn't exist. Or is that too confusing?
    """
    def __init__(self, dirname):
        """Create object linked to a data directory
        
        If directory does not exist, will create it.
        """
        self.full_path = os.path.normpath(dirname)
        self.session_name = os.path.basename(self.full_path)
        
        if not os.path.exists(self.full_path):
            os.mkdir(self.full_path)
    
    def get_db_filename(self):
        return os.path.join(self.full_path, self.session_name + '.db')
    
    def open_db(self):
        OE.open_db('sqlite:///' + self.get_db_filename())
    
    def get_OE_session(self):
        return OE.Session()
    
    def get_ns5_loader(self):
        """Returns Loader object for ns5 file"""
        l = ns5.Loader(filename=self.get_ns5_filename())
        l.load_file()
        return l
    
    def read_channel_groups(self):
        """Returns a list of channel groups"""
        return read_channel_numbers(\
            os.path.join(self.full_path, GROUPED_CHANNELS_FILENAME))
    
    def read_neural_channel_ids(self):
        """Returns a list of all channel numbers"""
        return read_channel_numbers(\
            os.path.join(self.full_path, ALL_CHANNELS_FILENAME))[0]
    
    def read_analog_channel_ids(self):
        """Returns a list of analog channels, or None if no file"""
        try:
            return read_channel_numbers(\
                os.path.join(self.full_path, ANALOG_CHANNELS_FILENAME))[0]
        except IOError:
            return None
    
    def read_time_limits(self):
        """Returns tuple of soft time limits and then hard, or None if missing"""
        try:
            data = read_channel_numbers(os.path.join(self.full_path,
                TIME_LIMITS_FILENAME), dtype=np.float)
        except IOError:
            return None
        
        return data[0], data[1]
    
    def write_time_limits(self, soft_time_limits, hard_time_limits):
        """Writes time limits in seconds to directory"""
        write_channel_numbers(os.path.join(self.full_path, 
            TIME_LIMITS_FILENAME), [soft_time_limits, hard_time_limits])

    def write_channel_groups(self, list_of_lists):
        """Writes metadata for channel groupings."""
        write_channel_numbers(\
            os.path.join(self.full_path, GROUPED_CHANNELS_FILENAME),
            list_of_lists)
    
    def write_neural_channel_ids(self, list_of_channels):
        """Writes list of channels to be processed."""
        write_channel_numbers(\
            os.path.join(self.full_path, ALL_CHANNELS_FILENAME),
            [list_of_channels])
    
    def write_analog_channel_ids(self, list_of_channels):
        """Writes analog channel numbers."""
        write_channel_numbers(\
            os.path.join(self.full_path, ANALOG_CHANNELS_FILENAME),
            [list_of_channels])
    
    def get_ns5_filename(self):
        """Returns ns5 filename in recording session"""
        filename_list = glob.glob(os.path.join(self.full_path, '*.ns5'))
        if len(filename_list) == 0:
            return None        
        
        if len(filename_list) > 1:
            print "warning: multiple ns5 files exist in %s" % self.full_path
        
        return filename_list[0]

    def get_raw_data_block(self):
        # Open database, get session
        self.open_db()
        session = self.get_OE_session()
        
        # Check to see whether data has already been added
        q = session.query(OE.Block).filter(OE.Block.name=='Raw Data')
        if q.count() > 0:
            return q.one()   
        else:
            return None
    
    def add_file(self, filename):
        """Copy file into RecordingSession."""
        shutil.copy(filename, self.full_path)
    
    def add_timestamps(self, list_of_values):
        """Adds timestamps by writing values to file in directory"""
        # different format, one value per line
        list_to_write = [[v] for v in list_of_values]        
        write_channel_numbers(\
            os.path.join(self.full_path, TIMESTAMPS_FILENAME),
            list_to_write)
    
    def read_timestamps(self):
        t = read_channel_numbers(os.path.join(self.full_path, 
            TIMESTAMPS_FILENAME))
        return np.array([tt[0] for tt in t])
    
    def put_neural_data_into_db(self, soft_limits_sec=None, 
        hard_limits_sec=None):
        """Loads neural data from ns5 file and puts into OE database.
        
        Slices around times provided in TIMESTAMPS. Also puts events in
        with label 'Timestamp' at each TIMESTAMP.
       
        Time limits will be loaded from disk unless provided.
        
        Returns OE block.
        """
        # Open database, get session
        self.open_db()
        session = self.get_OE_session()
        
        # See if operation already occurred
        block = self.get_raw_data_block()
        if block is not None:
            return block      
        
        # Read time stamps and set limits in samples
        t = self.read_timestamps()
        l = self.get_ns5_loader()
        
        # Get time limits for slicing
        if hard_limits_sec is None:
            hard_limits_sec = self.read_time_limits()[1]
        if soft_limits_sec is None:
            soft_limits_sec = self.read_time_limits()[0]
        hard_limits = np.array(\
            np.asarray(hard_limits_sec) * l.header.f_samp, dtype=np.int)
        soft_limits = np.array(\
            np.asarray(soft_limits_sec) * l.header.f_samp, dtype=np.int)
        
        # Slice Trials around timestamps
        t_starts, t_stops = TrialSlicer.slice_trials(\
            timestamps=t,
            soft_limits=soft_limits, 
            hard_limits=hard_limits, 
            meth='end_of_previous', 
            data_range=(0, l.header.n_samples))
        
        # Load data from file and store all neural channels in block
        blr = OE.neo.io.BlackrockIO(self.get_ns5_filename())        
        block = blr.read_block(full_range=FULL_RANGE_UV, t_starts=t_starts, 
            t_stops=t_stops, chlist=(self.read_neural_channel_ids() + 
            [ch+128 for ch in self.read_analog_channel_ids()]))
        
        # Convert to OE object
        block2 = OE.io.io.hierachicalNeoToOe(block)
        block2.name = 'Raw Data'

        # Add events at TIMESTAMPS
        if len(t) == len(block2._segments):
            for tt, seg in zip(t, block2._segments):
                e = OE.Event(name='Timestamp', label='Timestamp')
                e.time = (tt/l.header.f_samp)
                #e.save()
                seg._events.append(e)
        else:
            print "warning: timestamps were dropped so I can't add events"

        # Save to database
        block2.save(session=session)
        
        return block2
    

    
    def avg_over_list_of_events(self, event_list, chn, meth='avg', t_start=None, t_stop=None):
        """Given a list of Event and a channel number, returns average signal.
        
        event_list : list of OE Event, ordered by id_segment
        chn : channel number
        session : the OE session from which you acquired the list of events
        
        Currently it's required that the list of Event be sorted by
        id_segment. For some reason if I sort it here, it causes an error.
        It's also required that only no two Event come from the same Segment.
        
        All AnalogSignal from channel `chn` containing an Event in event_list
        will be averaged together, triggered on the time of the Event.
        
        This function grabs the signals, error-checks, and then
        the actual work is done by a lower level function that averages
        AnalogSignal (but does not error check).
        """
        # Extract segment id from each event and test for ordering
        # I get weird detached instance errors here some times, so replace:
        #seg_id_list = [e.segment.id for e in event_list]
        seg_id_list = [e.id_segment for e in event_list]
        assert seg_id_list == sorted(seg_id_list), "events must be sorted by segment"
        
        # Why doesn't this work?
        # This is what would fix the segment ordering restriction
        #idxs = np.argsort(seg_id_list)                
        #event_list = list(np.take(event_list, idxs))
        #seg_id_list = list(np.take(seg_id_list, idxs))
        
        # Get Segment containing this Event and check that there is only
        # one Event per Segment, so that we can use the ordering to ensure
        # one-to-one relationship between Event and AnalogSignal        
        if len(np.unique(seg_id_list)) != len(seg_id_list):
            raise(ValueError("More than one of the specified event per segment"))        
        
        # Get list of signals with this channel and in the list of Segment
        # again ordered by id_segment, so signal_list[n] came from
        # seg_id_list[n] which contains event[n]
        signal_list = self.get_OE_session().query(OE.AnalogSignal).\
            filter(OE.AnalogSignal.channel == chn).\
            filter(OE.AnalogSignal.id_segment.in_(seg_id_list)).\
            order_by(OE.AnalogSignal.id_segment).all()
        
        # Check alignment
        assert np.all(
            np.array([sig.id_segment for sig in signal_list]) ==
            np.array(seg_id_list)), "Mismatched segments and events, somehow"
        
        # Call signal averaging function
        return self.avg_over_signals_with_triggers(
            signal_list, [e.time for e in event_list], meth=meth, 
            t_start=t_start, t_stop=t_stop)
    
    def avg_over_signals_with_triggers(self, signal_list, trigger_times, 
        t_start=None, t_stop=None, meth='avg'):
        """Averages list of AnalogSignal triggered on times.
        
        signal_list : list of AnalogSignal to be averaged
        trigger_times : trigger times, one per AnalogSignal
        t_start, t_stop : the returned average will go from t_start to
            t_stop, relative to trigger time. If None, then the maximum
            amount of time will be returned, which is set by the overlap
            of the time series contained in each signal.
        
        You must ensure that the trigger times are lined up correctly
        with AnalogSignal before calling this function, and that all
        AnalogSignal contain sufficient data for the t_start and t_stop
        you specify.
        
        Currently this function contains some paranoid error checking
        of the OE AnalogSignal.time_slice functionality.
        
        Returns tuple (return_t, avgsig) where return_t is an array
        of times (relative to trigger) and avgsig is the averaged value
        at those times.
        """
        #f = plt.figure()
        #ax = f.add_subplot(111)
        # If no times are provided, figure out maximum overlap
        if t_start is None:
            # Furthest back we can go
            t_start = np.max([sig.t_start - trigger \
                for sig, trigger in zip(signal_list, trigger_times)])
        if t_stop is None:
            # Furthest forward we can go
            t_stop = np.min([sig.t()[-1] - trigger \
                for sig, trigger in zip(signal_list, trigger_times)])        
        
        # Get all the slices in a list, get the returned times
        slices = [ ]
        return_t = None
        
        # Iterate through signals
        for sig, trigger in zip(signal_list, trigger_times):
            #assert(sig.segment._events[0].time == trigger)
            # Get time slice around trigger time and append to list
            #ax.plot(sig.t() - trigger, sig.signal)
            slc = sig.time_slice(trigger + t_start, trigger + t_stop)
            slices.append(slc.signal)
            
            # Error check return_t for consistency (paranoid)
            if return_t is None:
                return_t = slc.t() - trigger
            else:
                # The return_t must not differ by more than one sampling period
                assert len(return_t) == len(slc.t())
                assert np.all(\
                    (slc.t() - trigger - return_t) < (1./slc.sampling_rate))
        
        # Average and return
        avgsig = np.mean(slices, axis=0)
        #ax.plot(return_t, avgsig, 'k', lw=4)
        if meth == 'avg':
            return (return_t, avgsig)
        elif meth =='all':
            return (return_t, np.array(slices))
