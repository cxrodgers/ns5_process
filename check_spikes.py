# Grabs spike times from db that were calculated from within OE

import OpenElectrophy as OE
import numpy as np
import matplotlib.pyplot as plt

db_name = '/home/chris/Public/20110401_CR13A_audresp_data/0327_002/datafile_CR_CR13A_110327_002.db'
#db_name = '/home/chris/Public/20110401_CR13A_audresp_data/0403_002/datafile_CR_CR13A_110403_002.db'
#db_name = '/home/chris/Public/20110401_CR13A_audresp_data/0329_002/datafile_CR_CR13A_110329_002.db'
OE.open_db(url=('sqlite:///%s' % db_name))    

# Load neurons
id_block = OE.sql('select block.id from block where block.name = \
    "CAR Tetrode Data"')[0][0]
id_neurons, = OE.sql('select neuron.id from neuron where neuron.id_block = \
    :id_block', id_block=id_block)

plt.figure()
bigger_spiketimes = np.array([])
for id_neuron in id_neurons:
    n = OE.Neuron().load(id_neuron)
    
    # Grab spike times from all trials (segments)
    big_spiketimes = np.concatenate(\
        [spiketrain.spike_times - spiketrain.t_start \
        for spiketrain in n._spiketrains])
    bigger_spiketimes = np.concatenate([bigger_spiketimes, big_spiketimes])
    
    # Compute histogram
    nh, x = np.histogram(big_spiketimes, bins=100)
    x = np.diff(x) + x[:-1]
    
    
    plt.plot(x, nh / float(len(n._spiketrains)) )
    #plt.title(('RP%d: N%d' % (n.id_recordingpoint, id_neuron)))
    
plt.show()
