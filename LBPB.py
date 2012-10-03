"""Various ways of dealing with stimulus names in LBPB task"""

short2long = {'lelo': 'LEFT+LOW', 'rilo': 'RIGHT+LOW', 'lehi': 'LEFT+HIGH',
    'rihi': 'RIGHT+HIGH'}

sns = list(range(1, 13))
stimnames = [
    'lo_pc_go', 'hi_pc_no', 'le_lc_go', 'ri_lc_no',
    'le_hi_pc', 'ri_hi_pc', 'le_lo_pc', 'ri_lo_pc',
    'le_hi_lc', 'ri_hi_lc', 'le_lo_lc', 'ri_lo_lc']
sn2name = {k: v for k, v in zip(sns, stimnames)}

mixed_stimnames = stimnames[4:]
mixed_sns = sns[4:]
mixed_sn2name = {k: v for k, v in zip(mixed_sns, mixed_stimnames)}

sound_block_tuple = (
    ('lehi', 'PB'), ('rihi', 'PB'), ('lelo', 'PB'), ('rilo', 'PB'),
    ('lehi', 'LB'), ('rihi', 'LB'), ('lelo', 'LB'), ('rilo', 'LB'))
block_sound_tuple = tuple(t[::-1] for t in sound_block_tuple)

stimname2sound_block_tuple = {sn: t 
    for sn, t in zip(mixed_stimnames, sound_block_tuple)}
stimname2block_sound_tuple = {sn: t 
    for sn, t in zip(mixed_stimnames, block_sound_tuple)}


sn2sound_block_tuple = {sn: t 
    for sn, t in zip(mixed_sns, sound_block_tuple)}
sn2block_sound_tuple = {sn: t 
    for sn, t in zip(mixed_sns, block_sound_tuple)}

mixed_stimnames_noblock = ('le_hi', 'ri_hi', 'le_lo', 'ri_lo')