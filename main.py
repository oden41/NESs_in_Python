import numpy as np
import random as rand
import experiments.experiment as ex
import methos.dx_nes

if __name__ == '__main__':
    opt = methos.dx_nes.DX_NES(20, lambda x : np.linalg.norm(x), 8, 3, 2)
    opt.do_oneiteration()