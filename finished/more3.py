"""Does the best case survive when the model reads 512 tokens?"""
import glob, os
import numpy as np
import domains as d
d.TRUST = d.TEST = 5000
D = sorted(x for x in glob.glob(d.SITE + "transformers/models/*") if os.path.isdir(x)); np.random.default_rng(0).shuffle(D)
A, B, C = ([p for x in part for p in d.files(x + "/**/*.py")] for part in d.parts(D))
d.run("code transformers unseen folders context 512", A, B, C, BLOCK=512, BATCH=4, last=120)
