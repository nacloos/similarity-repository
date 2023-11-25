import numpy as np
import rtd

np.random.seed(7)
P = np.random.rand(1000, 2)
Q = np.random.rand(1000, 2)


barc = rtd.calc_embed_dist(P, Q)
rtd.plot_barcodes(rtd.barc2array(barc))

rtd.rtd(P, Q)
