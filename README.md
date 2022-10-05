# Gesture


- code in `src`: run gesture model (choose between LSTM and IMV-LSTM)

For example: to run IMV-LSTM with hyperparamter search, use `python main.py -m +hsearch=lr-auc-IMV data.use_k_fold=True module/model=TwoHeadIMVLSTM`

- `scripts/vis`: run visualization of gesture model

- `scripts/gesture_only`: run gesture model

- `scripts/clinic_only`: run clinical model

- `scripts/clinic+gesture`: run clinical + gesture model