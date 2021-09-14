# FaSNet-TAC-pyTorch
Full implementation of "End-to-end microphone permutation and number invariant multi-channel speech separation" (Interspeech 2020)



## Plan

- [x] Data pre-processing
- [x] Training
- [x] Inference
- [x] Separate


## How to use?
First, you have to generate dataset from followed link.

Data generation script: https://github.com/yluo42/TAC/tree/master/data

You can use our code by changing data_script/tr.scp, cv.scp, tt.scp as your data directory.

```bash
# In scp file

D:/MC_Libri_fixed/tr # your path
20000 # the number of samples
```

Second, to train the model use 
```bash
python train.py
```

Third, to evaluate metrics on test set use
```bash
python evaluate.py
```

Fourth, to separate test set to speakers use
```bash
python separate.py
```

## Reference
https://github.com/yluo42/TAC/

## Result

We achieved SI-SNRi 11.36 dB in 6 microphone noisy reverberant setting.
