
class KK_loader(object):
    """Loads spike info from KlustaKwik-compatible files."""
    def __init__(self, data_dir):
        """Initializes a new KK_loader, linked to the data directory.
        
        data_dir : string, path to location of KlustaKwik files.
        """
        self._data_dir = data_dir

    def execute(self):
        """Loads spike times and unit info from KlustaKwik files.
        
        Stores resulting spike trains in self.spiketrains, a dict keyed
        by tetrode number.
        """
        # Search data directory for KlustaKwik files. This sets
        # self._clufiles and self._fetfiles.
        self._get_fns_from_datadir()
        
        # Load spike times from each tetrode and store in a dict, keyed
        # by tetrode number
        self.spiketrains = dict()
        for ntet in self._fetfiles.keys():
            fetfile = self._fetfiles[ntet]
            clufile = self._clufiles[ntet]            
            spks = self._load_spike_times(fetfile)
            uids = self._load_unit_id(clufile)
            
            # Initialize a new container for the loaded spike times and IDs
            self.spiketrains[ntet] = MultipleUnitSpikeTrain(spks, uids)
    
    def _get_fns_from_datadir(self):
        """Stores dicts of KlustaKwik *.clu and *.fet files from directory."""
        self._clufiles = self._parse_KK_files(\
            glob.glob(os.path.join(self._data_dir, '*.clu.*')), 'clu')
        self._fetfiles = self._parse_KK_files(\
            glob.glob(os.path.join(self._data_dir, '*.fet.*')), 'fet')

    def _parse_KK_files(self, filename_list, match_string='fet'):
        """Returns a dict of filenames keyed on tetrode number.
        
        You give a list of filenames, obtained by globbing, that are either
        cluster or feature files. You also specify whether they are `clu`
        or `fet` files.
        
        This matches on tetrode number and returns a dict of the filenames
        keyed by tetrode number.
        
        It must end with digits, not including minus sign or anything else.
        """
        d = dict()
        for v in filename_list:
            # Test whether matches format, ie ends with digits
            m = glob.re.search(('%s\.(\d+)' % match_string), v)             
            if m is not None:
                # Key the tetrode number to the filename
                tetn = int(m.group(1)) # Group 0 is something else
                d[tetn] = v
        return d
        
        #~ return dict([\
            #~ (int(glob.re.search(('%s\.(\d+)' % match_string), v).group(1)), v) \
            #~ for v in filename_list])

    def _load_spike_times(self, fetfilename):
        f = file(fetfilename, 'r')
        
        # Number of clustering features is integer on first line
        nbFeatures = int(f.readline().strip())
        
        # Each subsequent line consists of nbFeatures values, followed by
        # the spike time in samples.
        names = ['feat%d' % n for n in xrange(nbFeatures)]
        names.append('spike_time')
        
        # Load into recarray
        data = mlab.csv2rec(f, names=names, skiprows=1)
        f.close()
        
        # Return the spike_time column
        return data['spike_time']

    def _load_unit_id(self, clufilename):
        f = file(clufilename, 'r')
        
        # Number of clusters on this tetrode is integer on first line
        nbClusters = int(f.readline().strip())
        
        # Each subquent line is a cluster ID (string)
        cluster_names = f.readlines()
        f.close()

        # Extract the number of the OE neuron name        
        #~ cluster_ids = np.array([\
            #~ int(glob.re.match('Neuron (\d+) ', name).group(1)) \
            #~ for name in cluster_names])
        cluster_ids = np.zeros((len(cluster_names),))
        for n, name in enumerate(cluster_names):
            m = glob.re.match('Neuron (\d+) ', name)
            if m is not None:
                cluster_ids[n] = int(m.group(1))
            else:
                cluster_ids[n] = 99

        # Simple error checking
        assert(len(np.unique(cluster_ids)) == nbClusters)
        assert(len(np.unique(cluster_names)) == nbClusters)

        return cluster_ids

