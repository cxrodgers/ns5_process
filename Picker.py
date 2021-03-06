from __future__ import print_function
from __future__ import division
from builtins import zip
from past.utils import old_div
from builtins import object
import numpy as np
import matplotlib.mlab as mlab
from functools import reduce

"""TODO:
add __getitem__ to call self._data.__getitem__
"""


class Picker(object):
    def __init__(self, data, method=1, info="A Picker"):
        self._data = data
        self.info = info
        #~ self.method = method
        #~ if method is 1:
            #~ self._calculate_pick_mask = self._calculate_pick_mask_meth1
        #~ elif method is 2:
            #~ self._calculate_pick_mask = self._calculate_pick_mask_meth2
        #~ elif method is 3:
            #~ self._calculate_pick_mask = self._calculate_pick_mask_meth3
    
    def __getitem__(self, *args, **kwargs):
        # this should return a Picker
        return self._data.__getitem__(*args, **kwargs)
    
    def __len__(self):
        return self._data.__len__()
    
    def append(self, picker_list, **kwargs):
        """Resize my data and add in the data from Pickers in picker_list
        note: equality test fails on picker2 for some reason 
        
        Will also add a new column if you specify.
        
        Usage:
        p1.append([p2, p3], ratname=(1,2,3))
        
        Now p1 has all of the data from p1, p2, and p3.
        p1['ratname'] is 1, 2, or 3, depending on the source.        
        """
        # Calculate new size and resize
        old_length = len(self)
        new_length = old_length + np.sum(len(p) for p in picker_list)
        new_data = np.resize(self._data, (new_length,))
        
        # Store data from each new picker
        row_idx = old_length
        for picker in picker_list:
            new_data[row_idx:row_idx+len(picker._data)] = picker._data
            row_idx += len(picker._data)
        
        # optionally add a new column
        if len(kwargs) > 0:
            if len(kwargs) > 1: print("warning: too many arguments")
            
            # get the name of the new field and the labels for each Picker
            fieldname = list(kwargs.keys())[0]
            labels = kwargs[fieldname]
            
            # create the new column and store the labels for each Picker
            newcolumn = np.empty(shape=(new_length,), dtype=np.int)
            newcolumn[:old_length] = labels[0]
            row_idx = old_length
            for label, picker in zip(labels[1:], picker_list):
                newcolumn[row_idx:row_idx+len(picker._data)] = label
                row_idx += len(picker._data)
            
            # store the new column
            new_data = mlab.rec_append_fields(new_data, fieldname, newcolumn)
        
        # overwrite my data with the new version
        self._data = new_data
    
    def _get_cols_from_args(self, args, kwargs):
        """Parse calling syntax.
        
        If only a column name is provided, as a string, this is interpreted
        as selecting all values for that column. Otherwise, the kwargs are 
        (colname, ok_value_list). This function puts `args` into `kwargs`
        and returns.
        """
        for colname in args:
            kwargs[colname] = np.unique(self._data[colname])
        return kwargs
    
    def one(self, colname=None, *args, **kwargs):
        """Returns the one and only one result"""
        d = self._data[colname][self._calculate_pick_mask(**kwargs)]
        if len(d) != 1:
            raise ValueError("Requested only one, but %d exist" % len(d))
        return d[0]
    
    def __repr__(self):
        return self._data.__repr__()    
    
    def list_by(self, *args, **kwargs):
        """Returns a list of Picker split by colname.
        
        If multiple column names are provided, every possible combination
        is calculated and returned. Maybe what this should do is recursively
        apply keyword filters one by one instead, creating a tree.
        
        TODO: Under what circumstances would there be some natural ordering?
        Currently we just use np.unique on the values which sorts them.
        Is there some better way to do this? Often, the order in which the
        keys were inserted is a useful way to get the data back, but
        A) this is usually cheating and susceptible to bugs, and b) only
        works for last dimension.
        
        Note that this also restricts us to data types that np.unique can
        handle. Will that work for dtype=object?
        
        If there is never a natural ordering (of keys), then returning a list
        seems pointless. The data will always be ordered in its original
        ordering since it's extracted via a mask of booleans. However, if it's
        sliced and put back together, the order will probably change.
        """
        # Figure out what user wants to list by
        name_vals = self._get_cols_from_args(args, kwargs)

        # construct all combinations of parameters
        # ex: {c1: [1,2], c2: [3,4]} => [{c1:[1], c2:[3]}, {c1:[1], c2:[4]},
        #   {c1:[2], c2:[3]}, {c1:[2], c2:[4]}]
        f = lambda l, val: reduce(list.__add__, 
            [[ll + [xx] for xx in val] for ll in l])
        all_combos = reduce(f, list(name_vals.values()), [[]])
        
        # Slice Picker once for each combo
        p = list()
        for combo in all_combos:
            # Create a filter based on each colname and its value in this combo
            kwargs2 = dict([(k, [v]) for k, v in zip(list(name_vals.keys()), combo)])
            
            res = self.filter(**kwargs2)
            if len(res) > 0: 
                p.append(res)
            #p.append(Picker(self._data[self._calculate_pick_mask(**kwargs2)],
            #    info='sliced'))
        return p
    
    def __str__(self):
        return self.info
    
    def dict_by(self, *args, **kwargs):
        """Returns a dict of Picker split by colname"""
        # Figure out what user wants to list by
        name_vals = self._get_cols_from_args(args, kwargs)
        
        # Use only first value for now.
        # TODO: construct all combinations
        colname, vals = list(name_vals.items())[0]        
        
        # Slice Picker once for each
        p = dict()
        for val in vals:
            kwargs = dict(((colname, [val]),))
            p[val] = self._data[self._calculate_pick_mask(**kwargs)]
        return p
    
    def filter(self, **kwargs):
        """Keep only records matching certain conditions.
        
        TODO: store a link somehow to allow unfiltering?        
        """
        return Picker(self._data[self._calculate_pick_mask(**kwargs)])
    
    def pick_data(self, colname, **kwargs):
        """TODO: default to last column"""
        return self._data[colname][self._calculate_pick_mask(**kwargs)]
    
    def pick_mask(self, **kwargs):
        """TODO: memoize (optionally)"""
        return self._calculate_pick_mask(**kwargs)
    
    def _calculate_pick_mask(self, **kwargs):
        # Begin with all true
        mask = np.ones(self._data.shape, dtype=bool)
        
        for colname, ok_value_list in list(kwargs.items()):
            # OR together all records with _data['colname'] in ok_value_list
            one_col_mask = np.zeros_like(mask)
            try:
                for ok_value in ok_value_list:
                    one_col_mask = one_col_mask | (self._data[colname] == ok_value)
            except TypeError:
                raise TypeError("ok value list must be iterable")
            # AND together the full mask with the results from this column
            mask = mask & one_col_mask
        
        return mask

    def _calculate_pick_mask_meth1B(self, kwargs):
        # Begin with all true
        mask = np.ones(self._data.shape, dtype=bool)
        
        for colname, ok_value_list in list(kwargs.items()):
            # OR together all records with _data['colname'] in ok_value_list
            one_col_mask = np.zeros_like(mask)            
            for ok_value in ok_value_list:
                one_col_mask |= (self._data[colname] == ok_value)
            
            # AND together the full mask with the results from this column
            mask &= one_col_mask
        
        return mask
    
    def _calculate_pick_mask_meth2(self, kwargs):
        mask = reduce(np.logical_and, 
                        [reduce(np.logical_or, 
                                [self._data[colname] == ok_value
                                    for ok_value in ok_value_list]) \
                            for colname, ok_value_list in list(kwargs.items())])
        
        return mask
    
    def _calculate_pick_mask_meth3(self, kwargs):
        # Begin with all true
        mask = np.ones(self._data.shape, dtype=bool)

        for colname, ok_value_list in list(kwargs.items()):
            # Find rows where data['colname'] is in ok_value_list
            one_col_mask = np.array([t in ok_value_list for t in \
                self._data[colname]])            
            
            # AND together the full mask with the results from this column
            mask = mask & one_col_mask
        
        return mask  

    def _calculate_pick_mask_meth4(self, kwargs):
        mask = reduce(np.logical_and, 
                        [np.array([t in ok_value_list for t in \
                                    self._data[colname]]) \
                            for colname, ok_value_list in list(kwargs.items())])
        
        return mask

if __name__ is '__main__':
    # Generate some fake data
    N_RECORDS = 1000000
    N_TRIALS = 500
    N_UNITS = 5
    x = np.recarray(shape=(N_RECORDS,),
        dtype=[('unit_id', int), ('trial_id', int), ('spike_time', float)])
    x['spike_time'] = np.random.random(N_RECORDS)
    x['trial_id'] = (np.random.random(N_RECORDS)*N_TRIALS).astype(int)
    x['unit_id'] = (np.random.random(N_RECORDS)*N_UNITS).astype(int)
    
    #~ x = np.recarray(shape=(10,),
        #~ dtype=[('col1', int), ('col2', int), ('col3', float)])
    
    
    # Build a picker
    p = Picker(data=x, method=1)
    pick_trials = np.arange(old_div(N_TRIALS,2))
    pick_units = np.arange(old_div(N_UNITS,2))
    print(p.pick_data('spike_time', trial_id=pick_trials, unit_id=pick_units).sum())
    
    # Nest test
    # prints data arrays arranged by unit_id
    for pp in p.list_by('unit_id'):
        print(pp._data)
    
    # prints data arrays for unit 1, then 2, then 3
    for pp in p.list_by(unit_id=[1,2,3]):
        print(pp._data)
    
    # prints n, then data array with unit_id==n, unordered
    for unit_id, pp in p.dict_by(unit_id=[1,2,3]):
        print(unit_id)
        print(pp._data)
    
    # add another session
    p.add(q, block=(1,2))
    
    # now p has original data (block=1) and data in q (block=2)
    # add another session
    p.add(r, block=3)
    
    # List by unit, block = (1,1); (1,2); (1,3); (2,1); etc
    p.list_by(unit_id=[1,2,3], block=[1,2,3])
    
    p.filter(block=1).list_by('unit_id').take('spike_times')
    # or
    [pp.take('spike_times') for pp in p.filter(block=1).list_by('unit_id')]