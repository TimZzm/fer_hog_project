"""FER2013 HOG emotion recognition project."""



"""
~/miniforge3/envs/dedalus22/bin/python run_pipeline.py \
  --csv ../fer2013.csv \
  --implementation baseline \
  --classifier logreg \
  --log-file logreg_baseline.log

~/miniforge3/envs/dedalus22/bin/python run_pipeline.py \
  --csv ../fer2013.csv \
  --implementation numpy \
  --classifier logreg \
  --log-file logreg_numpy.log

~/miniforge3/envs/dedalus22/bin/python run_pipeline.py \
  --csv ../fer2013.csv \
  --implementation numpy_numba \
  --classifier logreg \
  --log-file logreg_numpy_numba.log

~/miniforge3/envs/dedalus22/bin/python run_pipeline.py \
  --csv ../fer2013.csv \
  --implementation numpy_parallel \
  --classifier logreg \
  --log-file logreg_numpy_parallel.log

~/miniforge3/envs/dedalus22/bin/python run_pipeline.py \
  --csv ../fer2013.csv \
  --implementation numpy_numba_parallel \
  --classifier logreg \
  --log-file logreg_numpy_numba_parallel.log
"""