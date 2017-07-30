import pandas as pd
import numpy as np
from model_mix_greedy2 import ModelMixGreedy2
from model import ModelRunnerBase
import copy
import h5py
from util import Util
from submission import Submission
from run_mix_greedy2 import RunnerMixGreedy2

def runmix(name, flist):
    runner = RunnerMixGreedy2(name, flist=flist, prms={})
    runner.run_train()
    runner.run_test()
    Submission.make_submission(name)

if __name__ == "__main__":

    mix_fin1_greedy6 = ["mix_fin1_greedy3", "mix_fin1_stack1"]
    runmix("mix_fin1_greedy6", mix_fin1_greedy6)