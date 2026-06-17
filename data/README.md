# Data

This project uses the scaled LIBSVM copy of the Pima Indians diabetes dataset:

https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes_scale

The raw data file is not committed to this repository. Run the experiment script
from the repository root and it will download `data/diabetes_scale` if needed:

```bash
python scripts/run_experiment.py
```

Label note: the LIBSVM file stores the diabetes-positive minority class as raw
label `-1`. The experiment maps raw label `-1` to `diabetes_positive=1` and raw
label `+1` to `diabetes_positive=0`.
