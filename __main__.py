import sys
import nnexec.TrainDecayMode as trainer
import nnexec.TuneDecayMode as tuner

mode = sys.argv[1]
cfg = sys.argv[2]

if mode == "Train":
    trainer.train(cfg)
elif mode == "Tune":
    tuner.tune(cfg)
else:
    raise ValueError()
