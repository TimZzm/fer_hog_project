from fer_hog.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())


"""
sample bash
~/miniforge3/envs/dedalus22/bin/python run_pipeline.py \
  --csv ../fer2013.csv \
  --implementation numpy \
  --log-file numba_run.log
"""