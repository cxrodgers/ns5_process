import OpenElectrophy as OE
import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path
import matplotlib
matplotlib.rcParams['figure.subplot.hspace'] = .5
matplotlib.rcParams['figure.subplot.wspace'] = .5
matplotlib.rcParams['font.size'] = 8.0
matplotlib.rcParams['xtick.labelsize'] = 'small'
matplotlib.rcParams['ytick.labelsize'] = 'small'



def run(db_name, save_fig_name=None):
    """If save_fig_name is not None, should be ex. '~/test.png' """
    # Load the raw data
    db = OE.open_db(url=('sqlite:///%s' % db_name))
    id_blocks, = OE.sql('SELECT block.id FROM block WHERE block.name="Raw Data"')
    id_block = id_blocks[0]

    id_recordingpoints, rp_names = OE.sql("SELECT \
        recordingpoint.id, recordingpoint.name \
        FROM recordingpoint \
        WHERE recordingpoint.id_block = :id_block", id_block=id_block)

    f = plt.figure(figsize=(10,10))
    
    # Process each recording point separately
    for n, (id_rp,tt) in enumerate(zip(id_recordingpoints[:16], rp_names[:16])):
        # Load all signals from all segments with this recording point
        id_sigs, = OE.sql('SELECT analogsignal.id FROM analogsignal ' + \
            'WHERE analogsignal.id_recordingpoint = :id_recordingpoint',
            id_recordingpoint=id_rp)
        
        # Average the signal
        avgsig = np.zeros(OE.AnalogSignal().load(id_sigs[0]).signal.shape)
        for id_sig in id_sigs: 
            sig = OE.AnalogSignal().load(id_sig)
            avgsig = avgsig + sig.signal
        avgsig = avgsig / len(id_sigs)

        # Plot the average signal of this recording point
        ax = f.add_subplot(4,4,n+1)
        ax.plot(np.arange(len(avgsig)) / sig.sampling_rate * 1000, avgsig)
        #ax.set_ylim((-250, 250))
        ax.set_title(tt)

    if save_fig_name is None:
        plt.show()
    else:
        plt.savefig(save_fig_name)
        plt.close()

if __name__ == '__main__':
    fn = sys.argv[1]
    print fn
    run(fn)