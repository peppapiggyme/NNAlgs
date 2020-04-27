import sys
import nnexec as ExecModule

mode = sys.argv[1]
cfg = sys.argv[2]
myExec = getattr(ExecModule, sys.argv[3])

if mode == "Train":
    myExec.train(cfg)
elif mode == "Tune":
    myExec.tune(cfg)
else:
    raise ValueError()
