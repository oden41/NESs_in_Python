import numpy as np
import random as rand
import experiments.experiment as ex
import methods.dx_nes

if __name__ == '__main__':
    opt = methods.dx_nes.DX_NES(20, lambda x : np.linalg.norm(x) ** 2, 8, 3, 2)
    opt.do_oneiteration()
    while opt.best_eval > 1e-9 and opt.eval_count < 1e5:
        opt.do_oneiteration()
    print(opt.eval_count)
    print(opt.best_eval)