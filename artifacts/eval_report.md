# RelayGuard evaluation report

- split: `test`
- samples: 436

## Detector: `gbm`

| auc | eer | hit_at_fpr_1 | hit_at_fpr_2 | hit_at_fpr_5 |
|---|---|---|---|---|
| 0.9842 | 0.0489 | 0.6375 | 0.8250 | 0.9500 |

Operating threshold @ FPR 2%: `0.6553`

### By metadata label
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| direct | 80 | 0 | - | 0.0375 | 0.9730 |
| hardneg_headset | 36 | 0 | - | 0.0000 | 0.9878 |
| hardneg_ns | 80 | 0 | - | 0.0000 | 0.9958 |
| hardneg_reverb | 80 | 0 | - | 0.0375 | 0.9722 |
| hardneg_tv | 80 | 0 | - | 0.0000 | 0.9941 |
| relay | 80 | 80 | 0.8250 | - | - |


### By codec2
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm | 15 | 15 | 1.0000 | - | - |
| mulaw | 10 | 10 | 0.9000 | - | - |
| none | 356 | 0 | - | 0.0169 | 0.9842 |
| opus | 55 | 55 | 0.7636 | - | - |


### By codec pair (codec1->codec2)
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm->gsm | 6 | 6 | 1.0000 | - | - |
| gsm->mulaw | 3 | 3 | 1.0000 | - | - |
| gsm->none | 106 | 0 | - | 0.0189 | 0.9765 |
| gsm->opus | 42 | 42 | 0.7857 | - | - |
| mulaw->gsm | 3 | 3 | 1.0000 | - | - |
| mulaw->mulaw | 3 | 3 | 1.0000 | - | - |
| mulaw->none | 127 | 0 | - | 0.0315 | 0.9796 |
| mulaw->opus | 7 | 7 | 0.8571 | - | - |
| opus->gsm | 6 | 6 | 1.0000 | - | - |
| opus->mulaw | 4 | 4 | 0.7500 | - | - |
| opus->none | 123 | 0 | - | 0.0000 | 0.9954 |
| opus->opus | 6 | 6 | 0.5000 | - | - |


### By device preset
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| bluetooth_mini | 20 | 20 | 0.9500 | - | - |
| budget_android | 8 | 8 | 1.0000 | - | - |
| car_speaker | 10 | 10 | 0.5000 | - | - |
| headset | 36 | 0 | - | 0.0000 | 0.9878 |
| iphone_earpiece | 7 | 7 | 1.0000 | - | - |
| laptop_speaker | 8 | 8 | 0.6250 | - | - |
| none | 320 | 0 | - | 0.0187 | 0.9838 |
| pixel_speaker | 14 | 14 | 0.7143 | - | - |
| tablet | 7 | 7 | 1.0000 | - | - |
| watch_speaker | 6 | 6 | 0.8333 | - | - |


## Detector: `cnn`

| auc | eer | hit_at_fpr_1 | hit_at_fpr_2 | hit_at_fpr_5 |
|---|---|---|---|---|
| 0.9977 | 0.0139 | 0.9750 | 0.9750 | 0.9750 |

Operating threshold @ FPR 2%: `0.5797`

### By metadata label
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| direct | 80 | 0 | - | 0.0000 | 0.9977 |
| hardneg_headset | 36 | 0 | - | 0.0000 | 0.9958 |
| hardneg_ns | 80 | 0 | - | 0.0000 | 0.9992 |
| hardneg_reverb | 80 | 0 | - | 0.0125 | 0.9952 |
| hardneg_tv | 80 | 0 | - | 0.0000 | 0.9995 |
| relay | 80 | 80 | 0.9750 | - | - |


### By codec2
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm | 15 | 15 | 1.0000 | - | - |
| mulaw | 10 | 10 | 1.0000 | - | - |
| none | 356 | 0 | - | 0.0028 | 0.9977 |
| opus | 55 | 55 | 0.9636 | - | - |


### By codec pair (codec1->codec2)
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm->gsm | 6 | 6 | 1.0000 | - | - |
| gsm->mulaw | 3 | 3 | 1.0000 | - | - |
| gsm->none | 106 | 0 | - | 0.0094 | 0.9965 |
| gsm->opus | 42 | 42 | 0.9762 | - | - |
| mulaw->gsm | 3 | 3 | 1.0000 | - | - |
| mulaw->mulaw | 3 | 3 | 1.0000 | - | - |
| mulaw->none | 127 | 0 | - | 0.0000 | 0.9984 |
| mulaw->opus | 7 | 7 | 0.8571 | - | - |
| opus->gsm | 6 | 6 | 1.0000 | - | - |
| opus->mulaw | 4 | 4 | 1.0000 | - | - |
| opus->none | 123 | 0 | - | 0.0000 | 0.9980 |
| opus->opus | 6 | 6 | 1.0000 | - | - |


### By device preset
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| bluetooth_mini | 20 | 20 | 1.0000 | - | - |
| budget_android | 8 | 8 | 1.0000 | - | - |
| car_speaker | 10 | 10 | 0.9000 | - | - |
| headset | 36 | 0 | - | 0.0000 | 0.9958 |
| iphone_earpiece | 7 | 7 | 1.0000 | - | - |
| laptop_speaker | 8 | 8 | 0.8750 | - | - |
| none | 320 | 0 | - | 0.0031 | 0.9979 |
| pixel_speaker | 14 | 14 | 1.0000 | - | - |
| tablet | 7 | 7 | 1.0000 | - | - |
| watch_speaker | 6 | 6 | 1.0000 | - | - |


## Detector: `bandwidth_forensics`

| auc | eer | hit_at_fpr_1 | hit_at_fpr_2 | hit_at_fpr_5 |
|---|---|---|---|---|
| 0.5367 | 0.5607 | 0.1250 | 0.1375 | 0.2000 |

Operating threshold @ FPR 2%: `0.0663`

### By metadata label
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| direct | 80 | 0 | - | 0.0125 | 0.5767 |
| hardneg_headset | 36 | 0 | - | 0.0000 | 0.4611 |
| hardneg_ns | 80 | 0 | - | 0.0125 | 0.5217 |
| hardneg_reverb | 80 | 0 | - | 0.0375 | 0.5181 |
| hardneg_tv | 80 | 0 | - | 0.0000 | 0.5641 |
| relay | 80 | 80 | 0.1375 | - | - |


### By codec2
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm | 15 | 15 | 0.4000 | - | - |
| mulaw | 10 | 10 | 0.3000 | - | - |
| none | 356 | 0 | - | 0.0140 | 0.5367 |
| opus | 55 | 55 | 0.0364 | - | - |


### By codec pair (codec1->codec2)
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm->gsm | 6 | 6 | 0.5000 | - | - |
| gsm->mulaw | 3 | 3 | 0.3333 | - | - |
| gsm->none | 106 | 0 | - | 0.0094 | 0.3506 |
| gsm->opus | 42 | 42 | 0.0476 | - | - |
| mulaw->gsm | 3 | 3 | 0.3333 | - | - |
| mulaw->mulaw | 3 | 3 | 0.3333 | - | - |
| mulaw->none | 127 | 0 | - | 0.0315 | 0.3690 |
| mulaw->opus | 7 | 7 | 0.0000 | - | - |
| opus->gsm | 6 | 6 | 0.3333 | - | - |
| opus->mulaw | 4 | 4 | 0.2500 | - | - |
| opus->none | 123 | 0 | - | 0.0000 | 0.8701 |
| opus->opus | 6 | 6 | 0.0000 | - | - |


### By device preset
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| bluetooth_mini | 20 | 20 | 0.1500 | - | - |
| budget_android | 8 | 8 | 0.2500 | - | - |
| car_speaker | 10 | 10 | 0.0000 | - | - |
| headset | 36 | 0 | - | 0.0000 | 0.4611 |
| iphone_earpiece | 7 | 7 | 0.5714 | - | - |
| laptop_speaker | 8 | 8 | 0.0000 | - | - |
| none | 320 | 0 | - | 0.0156 | 0.5452 |
| pixel_speaker | 14 | 14 | 0.0714 | - | - |
| tablet | 7 | 7 | 0.0000 | - | - |
| watch_speaker | 6 | 6 | 0.1667 | - | - |


## Detector: `reverb`

| auc | eer | hit_at_fpr_1 | hit_at_fpr_2 | hit_at_fpr_5 |
|---|---|---|---|---|
| 0.7837 | 0.2875 | 0.0250 | 0.1750 | 0.3000 |

Operating threshold @ FPR 2%: `0.8038`

### By metadata label
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| direct | 80 | 0 | - | 0.0000 | 0.8372 |
| hardneg_headset | 36 | 0 | - | 0.0000 | 0.8878 |
| hardneg_ns | 80 | 0 | - | 0.0000 | 0.8850 |
| hardneg_reverb | 80 | 0 | - | 0.0625 | 0.5312 |
| hardneg_tv | 80 | 0 | - | 0.0000 | 0.8347 |
| relay | 80 | 80 | 0.1750 | - | - |


### By codec2
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm | 15 | 15 | 0.1333 | - | - |
| mulaw | 10 | 10 | 0.5000 | - | - |
| none | 356 | 0 | - | 0.0140 | 0.7837 |
| opus | 55 | 55 | 0.1273 | - | - |


### By codec pair (codec1->codec2)
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm->gsm | 6 | 6 | 0.1667 | - | - |
| gsm->mulaw | 3 | 3 | 0.3333 | - | - |
| gsm->none | 106 | 0 | - | 0.0283 | 0.7528 |
| gsm->opus | 42 | 42 | 0.0952 | - | - |
| mulaw->gsm | 3 | 3 | 0.0000 | - | - |
| mulaw->mulaw | 3 | 3 | 0.3333 | - | - |
| mulaw->none | 127 | 0 | - | 0.0079 | 0.7965 |
| mulaw->opus | 7 | 7 | 0.4286 | - | - |
| opus->gsm | 6 | 6 | 0.1667 | - | - |
| opus->mulaw | 4 | 4 | 0.7500 | - | - |
| opus->none | 123 | 0 | - | 0.0081 | 0.7973 |
| opus->opus | 6 | 6 | 0.0000 | - | - |


### By device preset
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| bluetooth_mini | 20 | 20 | 0.3000 | - | - |
| budget_android | 8 | 8 | 0.1250 | - | - |
| car_speaker | 10 | 10 | 0.2000 | - | - |
| headset | 36 | 0 | - | 0.0000 | 0.8878 |
| iphone_earpiece | 7 | 7 | 0.0000 | - | - |
| laptop_speaker | 8 | 8 | 0.1250 | - | - |
| none | 320 | 0 | - | 0.0156 | 0.7720 |
| pixel_speaker | 14 | 14 | 0.1429 | - | - |
| tablet | 7 | 7 | 0.1429 | - | - |
| watch_speaker | 6 | 6 | 0.1667 | - | - |


## Detector: `distortion`

| auc | eer | hit_at_fpr_1 | hit_at_fpr_2 | hit_at_fpr_5 |
|---|---|---|---|---|
| 0.8670 | 0.2275 | 0.2000 | 0.2375 | 0.4750 |

Operating threshold @ FPR 2%: `0.7927`

### By metadata label
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| direct | 80 | 0 | - | 0.0000 | 0.9264 |
| hardneg_headset | 36 | 0 | - | 0.0000 | 0.7149 |
| hardneg_ns | 80 | 0 | - | 0.0000 | 0.8781 |
| hardneg_reverb | 80 | 0 | - | 0.0750 | 0.8319 |
| hardneg_tv | 80 | 0 | - | 0.0125 | 0.9000 |
| relay | 80 | 80 | 0.2375 | - | - |


### By codec2
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm | 15 | 15 | 0.1333 | - | - |
| mulaw | 10 | 10 | 0.2000 | - | - |
| none | 356 | 0 | - | 0.0197 | 0.8670 |
| opus | 55 | 55 | 0.2727 | - | - |


### By codec pair (codec1->codec2)
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm->gsm | 6 | 6 | 0.1667 | - | - |
| gsm->mulaw | 3 | 3 | 0.6667 | - | - |
| gsm->none | 106 | 0 | - | 0.0094 | 0.8552 |
| gsm->opus | 42 | 42 | 0.3095 | - | - |
| mulaw->gsm | 3 | 3 | 0.0000 | - | - |
| mulaw->mulaw | 3 | 3 | 0.0000 | - | - |
| mulaw->none | 127 | 0 | - | 0.0236 | 0.8815 |
| mulaw->opus | 7 | 7 | 0.1429 | - | - |
| opus->gsm | 6 | 6 | 0.1667 | - | - |
| opus->mulaw | 4 | 4 | 0.0000 | - | - |
| opus->none | 123 | 0 | - | 0.0244 | 0.8622 |
| opus->opus | 6 | 6 | 0.1667 | - | - |


### By device preset
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| bluetooth_mini | 20 | 20 | 0.4000 | - | - |
| budget_android | 8 | 8 | 0.3750 | - | - |
| car_speaker | 10 | 10 | 0.1000 | - | - |
| headset | 36 | 0 | - | 0.0000 | 0.7149 |
| iphone_earpiece | 7 | 7 | 0.1429 | - | - |
| laptop_speaker | 8 | 8 | 0.0000 | - | - |
| none | 320 | 0 | - | 0.0219 | 0.8841 |
| pixel_speaker | 14 | 14 | 0.2857 | - | - |
| tablet | 7 | 7 | 0.1429 | - | - |
| watch_speaker | 6 | 6 | 0.1667 | - | - |


## Detector: `cnn_gbm_avg`

| auc | eer | hit_at_fpr_1 | hit_at_fpr_2 | hit_at_fpr_5 |
|---|---|---|---|---|
| 0.9965 | 0.0237 | 0.9250 | 0.9750 | 0.9875 |

Operating threshold @ FPR 2%: `0.4301`

### By metadata label
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| direct | 80 | 0 | - | 0.0375 | 0.9942 |
| hardneg_headset | 36 | 0 | - | 0.0000 | 0.9969 |
| hardneg_ns | 80 | 0 | - | 0.0000 | 0.9994 |
| hardneg_reverb | 80 | 0 | - | 0.0500 | 0.9938 |
| hardneg_tv | 80 | 0 | - | 0.0000 | 0.9984 |
| relay | 80 | 80 | 0.9750 | - | - |


### By codec2
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm | 15 | 15 | 1.0000 | - | - |
| mulaw | 10 | 10 | 1.0000 | - | - |
| none | 356 | 0 | - | 0.0197 | 0.9965 |
| opus | 55 | 55 | 0.9636 | - | - |


### By codec pair (codec1->codec2)
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm->gsm | 6 | 6 | 1.0000 | - | - |
| gsm->mulaw | 3 | 3 | 1.0000 | - | - |
| gsm->none | 106 | 0 | - | 0.0283 | 0.9948 |
| gsm->opus | 42 | 42 | 0.9524 | - | - |
| mulaw->gsm | 3 | 3 | 1.0000 | - | - |
| mulaw->mulaw | 3 | 3 | 1.0000 | - | - |
| mulaw->none | 127 | 0 | - | 0.0236 | 0.9963 |
| mulaw->opus | 7 | 7 | 1.0000 | - | - |
| opus->gsm | 6 | 6 | 1.0000 | - | - |
| opus->mulaw | 4 | 4 | 1.0000 | - | - |
| opus->none | 123 | 0 | - | 0.0081 | 0.9982 |
| opus->opus | 6 | 6 | 1.0000 | - | - |


### By device preset
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| bluetooth_mini | 20 | 20 | 1.0000 | - | - |
| budget_android | 8 | 8 | 1.0000 | - | - |
| car_speaker | 10 | 10 | 0.9000 | - | - |
| headset | 36 | 0 | - | 0.0000 | 0.9969 |
| iphone_earpiece | 7 | 7 | 1.0000 | - | - |
| laptop_speaker | 8 | 8 | 0.8750 | - | - |
| none | 320 | 0 | - | 0.0219 | 0.9964 |
| pixel_speaker | 14 | 14 | 1.0000 | - | - |
| tablet | 7 | 7 | 1.0000 | - | - |
| watch_speaker | 6 | 6 | 1.0000 | - | - |

