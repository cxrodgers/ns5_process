import numpy as np
import os.path
import scipy.io
import glob
import pickle

class Bcontrol_Loader_By_Dir(object):
    """Wrapper for Bcontrol_Loader to load/save from directory.
    
    Methods
    -------
    load : Get data from directory
    get_sn2trials : returns a dict of stimulus numbers and trial numbers
    get_sn2name : returns a dict of stimulus numbers and name
    """
    def __init__(self, dirname, auto_validate=True, v2_behavior=False,
        skip_trial_set=[]):
        """Initialize loader, specifying directory containing info.
        
        For other parameters, see Bcontrol_Loader
        """
        self.dirname = dirname
        self._pickle_name = 'bdata.pickle'
        self._bcontrol_matfilename = 'data_*.mat'
        
        # Build a Bcontrol_Loader with same parameters
        self._bcl = Bcontrol_Loader(auto_validate=auto_validate,
            v2_behavior=v2_behavior, skip_trial_set=skip_trial_set)
    
    def load(self):
        """Loads Bcontrol info into self.data.
        
        First checks to see if bdata pickle exists, in which case it loads
        that pickle. Otherwise, uses self._bcl to load data from matfile.
        """
        # Look for a pickle
        data, pickle_found = self._check_for_pickle()
        
        if pickle_found:
            self._bcl.data = data
        else:
            filename = self._find_bcontrol_matfile()
            self._bcl.filename = filename
            self._bcl.load()
            
            # Pickle self._bcl.data
            self._pickle_data()
        
        self.data = self._bcl.data
    
    def _check_for_pickle(self):
        """Tries to load bdata pickle if exists.
        
        Returns (data, True) if bdata pickle is found in self.dirname.
        Otherwise returns (None, False)
        """
        data = None
        possible_pickles = glob.glob(os.path.join(self.dirname, 
            self._pickle_name))
        if len(possible_pickles) == 1:
            # A pickle was found, load it
            f = file(possible_pickles[0], 'r')
            data = pickle.load(f)
            f.close()
        
        return (data, len(possible_pickles) == 1)
    
    def _find_bcontrol_matfile(self):
        """Returns filename to BControl matfile in self.dirname"""
        fn_bdata = glob.glob(os.path.join(self.dirname, 
            self._bcontrol_matfilename))
        assert(len(fn_bdata) == 1)
        return fn_bdata[0]
    
    def _pickle_data(self):
        """Pickles self._bcl.data for future use."""
        fn_pickle = os.path.join(self.dirname, self._pickle_name)
        f = file(fn_pickle, 'w')
        pickle.dump(self._bcl.data, f)
        f.close()
    
    def get_sn2trials(self, outcome='hit'):
        return self._bcl.get_sn2trials(outcome)
    
    def get_sn2names(self):
        return self._bcl.get_sn2names()
    
    def get_sn2name(self):
        return self._bcl.get_sn2names()


class Bcontrol_Loader(object):
    """Loads matlab BControl data and validates"""
    def __init__(self, filename=None, auto_validate=True, v2_behavior=False,
        skip_trial_set=[]):
        """Initialize loader, optionally specifying filename.
        
        If auto_validate is True, then the validation script will run
        after loading the data. In any case, you can always call the
        validation method manually.
        
        v2_behavior : boolean. If True, then looks for variables that
        work with TwoAltChoice_v2 (no datasink).
        
        skip_trial_set : list. Wherever TRIALS_INFO['TRIAL_NUMBER'] is
        a member of skip_trial_set, that trial will be skipped in the
        validation process.
        
        TODO: trigger v2_behavior automatically (with warning) if
        datasink does not exist
        """
        self.filename = filename
        self.auto_validate = auto_validate
        self.v2_behavior = v2_behavior
        self.skip_trial_set = np.array(skip_trial_set)
        
        # Set a variable for accessing TwoAltChoice_vx variable names
        if self.v2_behavior:
            self._vstring = 'v2'
        else:
            self._vstring = 'v4'
    
    
    def load(self, filename=None):
        """Loads the bcontrol matlab file.
        
        Loads the data from disk. Then, optionally, validates it. Finally,        
        returns a dict of useful information from the file, containing
        the following keys:
            TRIALS_INFO: a recarray of trial-by-trial info
            SOUNDS_INFO: describes the rules associated with each sound
            CONSTS: helps in decoding the integer values
            peh: the raw events and pokes from BControl
            datasink: debugging trial-by-trial snapshots
            onsets: the stimulus onsets, extracted from peh
        
        Note: for compatibility with Matlab, the stimulus numbers in
        TRIALS_INFO are numbered beginning with 1.
        """
        if filename is not None: self.filename = filename
        
        # Actually load the file from disk and store variables in self.data
        self._load()
        
        # Optionally, run validation
        # Will fail assertion if errors, otherwise you're fine
        if self.auto_validate: self.validate()
        
        # Return dict of import info
        return self.data
    
    def get_sn2trials(self, outcome='hit'):
        """Returns a dict: stimulus number -> trials on which it occurred.
        
        For each stimulus number, finds trials with that stimulus number
        that were not forced and with the specified outcome.
        
        Parameters
        ----------
        outcome : string. Will be tested against TRIALS_INFO['OUTCOME'].
        Should be hit, error, or wrong_port.
        
        Returns
        -------
        dict sn2trials, such that sn2trials[sn] is the list of trials on
        which sn occurred.
        """
        TRIALS_INFO = self.data['TRIALS_INFO']
        CONSTS = self.data['CONSTS']
        trial_numbers_vs_sn = dict()
        
        # Find all trials matching the requirements.
        for sn in np.unique(TRIALS_INFO['STIM_NUMBER']):
            keep_rows = \
                (TRIALS_INFO['STIM_NUMBER'] == sn) & \
                (TRIALS_INFO['OUTCOME'] == CONSTS[outcome.upper()]) & \
                (TRIALS_INFO['NONRANDOM'] == 0)        
            trial_numbers_vs_sn[sn] = TRIALS_INFO['TRIAL_NUMBER'][keep_rows]
        return trial_numbers_vs_sn  

    def get_sn2names(self):
        sn2name = dict([(n+1, sndname) for n, sndname in \
            enumerate(self.data['SOUNDS_INFO']['sound_name'])])
        return sn2name
    
    def _load(self):
        """Hidden method that actually loads matfile data and stores
        
        This is for low-level code that parse the BControl `saved`,
        `saved_history`, etc.
        """
        # Load the matlab file
        matdata = scipy.io.loadmat(self.filename, squeeze_me=True, 
            struct_as_record=False)
        saved = matdata['saved']
        saved_history = matdata['saved_history']

        # Load TRIALS_INFO matrix as recarray
        TRIALS_INFO = self._format_trials_info(saved)

        # Load CONSTS
        CONSTS = saved.__dict__[('TwoAltChoice_%s_CONSTS' % self._vstring)].\
            __dict__.copy()
        CONSTS.pop('_fieldnames')
        for (k,v) in CONSTS.items():
            try:
                # This will work if v is a 0d array (EPD loadmat)
                CONSTS[k] = v.flatten()[0]
            except AttributeError:
                # With other versions of loadmat, v is an int
                CONSTS[k] = v

        # Load SOUNDS_INFO
        SOUNDS_INFO = saved.__dict__[\
            ('TwoAltChoice_%s_SOUNDS_INFO' % self._vstring)].__dict__.copy()
        SOUNDS_INFO.pop('_fieldnames')

        # Now the trial-by-trial datasink, which does not exist in v2
        datasink = None
        if not self.v2_behavior:
            datasink = saved_history.__dict__[('TwoAltChoice_%s_datasink' % \
                self._vstring)]

        # And finally the stored behavioral events
        peh = saved_history.ProtocolsSection_parsed_events      

        # Extract out the parameter of most interest: stimulus onset
        onsets = np.array([trial.__dict__['states'].\
            __dict__['play_stimulus'][0] for trial in peh])        
        
        # Store
        self.data = dict((
            ('TRIALS_INFO', TRIALS_INFO),
            ('SOUNDS_INFO', SOUNDS_INFO),
            ('CONSTS', CONSTS),
            ('peh', peh),
            ('datasink', datasink),
            ('onsets', onsets)))


    def _format_trials_info(self, saved):
        """Hidden method to format TRIALS_INFO.

        Converts the matrix to a recarray and names it with the column
        names from TRIALS_INFO_COLS.
        """
        # Some constants that need to be converted from structs to dicts
        d2 = saved.__dict__[('TwoAltChoice_%s_TRIALS_INFO_COLS' % \
            self._vstring)].__dict__.copy()
        d2.pop('_fieldnames')   
        try:
            # This will work if loadmat returns 0d arrays (EPD)
            d3 = dict((v.flatten()[0], k) for k, v in d2.iteritems())
        except AttributeError:
            # With other versions, v is an int
            d3 = dict((v, k) for k, v in d2.iteritems())

        # Check that all the columns are named
        if len(d3) != len(d2):
            print "Multiple columns with same number in TRIALS_INFO_COLS"

        # Write the column names in order
        # Will error here if the column names are messed up
        # Note inherent conversion from 1-based to 0-based indexing
        field_names = [d3[col] for col in xrange(1,1+len(d3))]
        TRIALS_INFO = np.rec.fromrecords(\
            saved.__dict__[('TwoAltChoice_%s_TRIALS_INFO' % self._vstring)],
            titles=field_names)
        
        return TRIALS_INFO
    
    
    def validate(self):
        """Runs validation checks on the loaded data.
        
        There are unlimited consistency checks we could do, but only a few
        easy checks are implemented. The most problematic error would be
        inconsistent data in TRIALS_INFO, for example if the rows were
        written with the wrong trial number or something. That's the
        primary thing that is checkoed.
        
        It is assumed that we
        can trust the state machine states. So, the pokes are not explicitly
        checked to ensure the exact timing of behavioral events. This would
        be a good feature to add though. Instead, the indicator states are
        matched to TRIALS_INFO. When easy, I check that at least one poke
        in the right port occurred, but I don't check that it actually
        occurred in the window of opportunity.
        
        No block information is checked. This is usually pretty obvious
        if it's wrong.
        
        If there is a known problem on certain trials, set
        self.skip_trial_set to a list of trials to skip. Rows of
        TRIALS_INFO for which TRIAL_NUMBER matches a member of this set
        will be skipped (not validated).
        
        Checks:        
        1) Does the *_istate outcome match the TRIALS_INFO outcome
        2) For each possible trial outcome, the correct port must have
        been entered (or not entered).
        3) The stim number in TRIALS_INFO should match the other TRIALS_INFO
        characteristics in accordance with SOUNDS_INFO.
        4) Every trial in peh should be in TRIALS_INFO, all others should be
        FUTURE_TRIAL.
        """   
        # Shortcut references to save typing
        CONSTS = self.data['CONSTS']
        TRIALS_INFO = self.data['TRIALS_INFO']
        SOUNDS_INFO = self.data['SOUNDS_INFO']
        peh = self.data['peh']
        datasink = self.data['datasink']
        
        # Some inverse maps for looking up data in TRIALS_INFO
        outcome_map = dict((CONSTS[str.upper(s)], s) for s in \
            ('hit', 'error', 'wrong_port'))
        left_right_map = dict((CONSTS[str.upper(s)], s) for s in \
            ('left', 'right'))
        go_or_nogo_map = dict((CONSTS[str.upper(s)], s) for s in \
            ('go', 'nogo'))

        # Go through peh and for each trial, match data to TRIALS_INFO
        # Also match to datasink. Note that datasink is a snapshot taken
        # immediately before the next trial state machine was uploaded.
        # So it contains some information about previous trial and some
        # about next. It is also always length 1 more than peh
        for n, trial in enumerate(peh):
            # Skip trials
            if TRIALS_INFO['TRIAL_NUMBER'][n] in self.skip_trial_set:
                continue
            
            # Extract info from the current row of TRIALS_INFO
            outcome = outcome_map[TRIALS_INFO['OUTCOME'][n]]
            correct_side = left_right_map[TRIALS_INFO['CORRECT_SIDE'][n]]
            go_or_nogo = go_or_nogo_map[TRIALS_INFO['GO_OR_NOGO'][n]]
            
            # Note that we correct for 1- and 0- indexing into SOUNDS_INFO here
            stim_number = TRIALS_INFO['STIM_NUMBER'][n] - 1
            
            # TRIALS_INFO is internally consistent with sound parameters
            assert(TRIALS_INFO['CORRECT_SIDE'][n] == \
                SOUNDS_INFO['correct_side'][stim_number])
            assert(TRIALS_INFO['GO_OR_NOGO'][n] == \
                SOUNDS_INFO['go_or_nogo'][stim_number])
            
            # If possible, check datasink
            if not self.v2_behavior:
                # Check that datasink is consistent with TRIALS_INFO
                # First load the n and n+1 sinks, since the info is split
                # across them. The funny .item() syntax is because loading
                # Matlab structs sometimes produces 0d arrays.
                # This little segment of code is the only place where the
                # datasink is checked.
                prev_sink = datasink[n]
                next_sink = datasink[n+1]
                try:
                    assert(prev_sink.next_sound_id.stimulus.item() == \
                        TRIALS_INFO['STIM_NUMBER'][n])
                    assert(prev_sink.next_side.item() == \
                        TRIALS_INFO['CORRECT_SIDE'][n])
                    assert(prev_sink.next_trial_type.item() == \
                        TRIALS_INFO['GO_OR_NOGO'][n])
                    assert(next_sink.finished_trial_num.item() == \
                        TRIALS_INFO['TRIAL_NUMBER'][n])
                    assert(CONSTS[next_sink.finished_trial_outcome.item()] == \
                        TRIALS_INFO['OUTCOME'][n])
                except AttributeError:
                    # .item() syntax only required for some versions of scipy
                    assert(prev_sink.next_sound_id.stimulus == \
                        TRIALS_INFO['STIM_NUMBER'][n])
                    assert(prev_sink.next_side == \
                        TRIALS_INFO['CORRECT_SIDE'][n])
                    assert(prev_sink.next_trial_type == \
                        TRIALS_INFO['GO_OR_NOGO'][n])
                    assert(next_sink.finished_trial_num == \
                        TRIALS_INFO['TRIAL_NUMBER'][n])
                    assert(CONSTS[next_sink.finished_trial_outcome] == \
                        TRIALS_INFO['OUTCOME'][n])
            
            # Sound name is correct
            # assert(SOUNDS_INFO.sound_names[stim_number] == datasink[sound name]
            
            # Validate trial
            self._validate_trial(trial, outcome, correct_side, go_or_nogo)
        
        # All future trials should be marked as such
        # Under certain circumstances, TRIALS_INFO can contain information
        # about one more trial than peh. I think this is if the protocol
        # is turned off before the end of the trial.
        try:
            assert(np.all(TRIALS_INFO['OUTCOME'][len(peh):] == \
                CONSTS['FUTURE_TRIAL']))
        except AssertionError:
            print "warn: at least one more trial in TRIALS_INFO than peh."
            print "checking that it is no more than one ..."
            assert(np.all(TRIALS_INFO['OUTCOME'][len(peh)+1:] == \
                CONSTS['FUTURE_TRIAL']))

    
    def _validate_trial(self, trial, outcome, correct_side, go_or_nogo):
        """Dispatches to appropriate trial validation method"""
        # Check if *_istate matches TRIALS_INFO
        assert(trial.states.__dict__[outcome+'_istate'].size == 2)
        
        dispatch_table = dict((\
            (('hit', 'go'), self._validate_hit_on_go),
            (('error', 'go'), self._validate_error_on_go),
            (('hit', 'nogo'), self._validate_hit_on_nogo),
            (('error', 'nogo'), self._validate_error_on_nogo),
            (('wrong_port', 'go'), self._validate_wrong_port),
            (('wrong_port', 'nogo'), self._validate_wrong_port),
            ))
        
        validation_method = dispatch_table[(outcome, go_or_nogo)]
        validation_method(trial, outcome, correct_side)

    def _validate_hit_on_go(self, trial, outcome, correct_side):
        """For hits on go trials, rewarded side should match correct side
        And there should be at least one poke in correct side
        """
        assert(trial.states.__dict__[correct_side+'_reward'].size == 2)
        assert(trial.pokes.__dict__[str.upper(correct_side[0])].size > 0)
        assert(trial.states.hit_on_go.size == 2)
    
    def _validate_error_on_go(self, trial, outcome, correct_side):
        """For errors on go trials, the reward state should not be entered"""
        assert(trial.states.left_reward.size == 0)
        assert(trial.states.right_reward.size == 0)
        assert(trial.states.error_on_go.size == 2)
        
    def _validate_error_on_nogo(self, trial, outcome, correct_side):
        """For errors on nogo trials, no reward should have been delivered
        And at least one entry into the correct side
        """
        assert(trial.states.left_reward.size == 0)
        assert(trial.states.right_reward.size == 0)
        assert(trial.pokes.__dict__[str.upper(correct_side[0])].size > 0)
        assert(trial.states.error_on_nogo.size == 2)        
    
    def _validate_hit_on_nogo(self, trial, outcome, correct_side):
        """For hits on nogo trials, a very short reward state should have
        occurred (this is just how it is handled in the protocol)
        """
        assert(np.diff(trial.states.__dict__[correct_side+'_reward']) < .002)
        assert(trial.states.hit_on_nogo.size == 2)
    
    def _validate_wrong_port(self, trial, outcome, correct_side):
        """For wrong port trials, no reward state, and should have
        entered wrong side at least once
        """
        assert(trial.states.left_reward.size == 0)
        assert(trial.states.right_reward.size == 0)
        if correct_side == 'left': assert(trial.pokes.R.size > 0)
        else: assert(trial.pokes.L.size > 0)
