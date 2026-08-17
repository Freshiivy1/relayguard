# RelayGuard evaluation report

- split: `test`
- samples: 312

## Detector: `gbm`

| auc | eer | hit_at_fpr_1 | hit_at_fpr_2 | hit_at_fpr_5 |
|---|---|---|---|---|
| 0.9999 | 0.0032 | 0.9936 | 1.0000 | 1.0000 |

Operating threshold @ FPR 2%: `0.1112`

### By metadata label
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| real_direct | 156 | 0 | - | 0.0128 | 0.9999 |
| real_relay | 156 | 156 | 1.0000 | - | - |


### By codec2
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm | 56 | 56 | 1.0000 | - | - |
| mulaw | 52 | 52 | 1.0000 | - | - |
| none | 156 | 0 | - | 0.0128 | 0.9999 |
| opus | 48 | 48 | 1.0000 | - | - |


### By codec pair (codec1->codec2)
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm->gsm | 19 | 19 | 1.0000 | - | - |
| gsm->mulaw | 16 | 16 | 1.0000 | - | - |
| gsm->opus | 17 | 17 | 1.0000 | - | - |
| mulaw->gsm | 16 | 16 | 1.0000 | - | - |
| mulaw->mulaw | 20 | 20 | 1.0000 | - | - |
| mulaw->opus | 14 | 14 | 1.0000 | - | - |
| opus->gsm | 21 | 21 | 1.0000 | - | - |
| opus->mulaw | 16 | 16 | 1.0000 | - | - |
| opus->opus | 17 | 17 | 1.0000 | - | - |
| real->none | 156 | 0 | - | 0.0128 | 0.9999 |


### By device preset
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| bluetooth_mini | 25 | 25 | 1.0000 | - | - |
| budget_android | 12 | 12 | 1.0000 | - | - |
| car_speaker | 18 | 18 | 1.0000 | - | - |
| iphone_earpiece | 20 | 20 | 1.0000 | - | - |
| laptop_speaker | 19 | 19 | 1.0000 | - | - |
| pixel_speaker | 21 | 21 | 1.0000 | - | - |
| real_telephony | 156 | 0 | - | 0.0128 | 0.9999 |
| tablet | 23 | 23 | 1.0000 | - | - |
| watch_speaker | 18 | 18 | 1.0000 | - | - |


## Detector: `cnn`

| auc | eer | hit_at_fpr_1 | hit_at_fpr_2 | hit_at_fpr_5 |
|---|---|---|---|---|
| 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

Operating threshold @ FPR 2%: `0.5857`

### By metadata label
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| real_direct | 156 | 0 | - | 0.0000 | 1.0000 |
| real_relay | 156 | 156 | 1.0000 | - | - |


### By codec2
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm | 56 | 56 | 1.0000 | - | - |
| mulaw | 52 | 52 | 1.0000 | - | - |
| none | 156 | 0 | - | 0.0000 | 1.0000 |
| opus | 48 | 48 | 1.0000 | - | - |


### By codec pair (codec1->codec2)
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm->gsm | 19 | 19 | 1.0000 | - | - |
| gsm->mulaw | 16 | 16 | 1.0000 | - | - |
| gsm->opus | 17 | 17 | 1.0000 | - | - |
| mulaw->gsm | 16 | 16 | 1.0000 | - | - |
| mulaw->mulaw | 20 | 20 | 1.0000 | - | - |
| mulaw->opus | 14 | 14 | 1.0000 | - | - |
| opus->gsm | 21 | 21 | 1.0000 | - | - |
| opus->mulaw | 16 | 16 | 1.0000 | - | - |
| opus->opus | 17 | 17 | 1.0000 | - | - |
| real->none | 156 | 0 | - | 0.0000 | 1.0000 |


### By device preset
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| bluetooth_mini | 25 | 25 | 1.0000 | - | - |
| budget_android | 12 | 12 | 1.0000 | - | - |
| car_speaker | 18 | 18 | 1.0000 | - | - |
| iphone_earpiece | 20 | 20 | 1.0000 | - | - |
| laptop_speaker | 19 | 19 | 1.0000 | - | - |
| pixel_speaker | 21 | 21 | 1.0000 | - | - |
| real_telephony | 156 | 0 | - | 0.0000 | 1.0000 |
| tablet | 23 | 23 | 1.0000 | - | - |
| watch_speaker | 18 | 18 | 1.0000 | - | - |


## Detector: `bandwidth_forensics`

| auc | eer | hit_at_fpr_1 | hit_at_fpr_2 | hit_at_fpr_5 |
|---|---|---|---|---|
| 0.6417 | 0.3462 | 0.1667 | 0.2756 | 0.3718 |

Operating threshold @ FPR 2%: `0.0596`

### By metadata label
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| real_direct | 156 | 0 | - | 0.0192 | 0.6417 |
| real_relay | 156 | 156 | 0.2756 | - | - |


### By codec2
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm | 56 | 56 | 0.3393 | - | - |
| mulaw | 52 | 52 | 0.4615 | - | - |
| none | 156 | 0 | - | 0.0192 | 0.6417 |
| opus | 48 | 48 | 0.0000 | - | - |


### By codec pair (codec1->codec2)
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm->gsm | 19 | 19 | 0.2632 | - | - |
| gsm->mulaw | 16 | 16 | 0.5000 | - | - |
| gsm->opus | 17 | 17 | 0.0000 | - | - |
| mulaw->gsm | 16 | 16 | 0.3125 | - | - |
| mulaw->mulaw | 20 | 20 | 0.7000 | - | - |
| mulaw->opus | 14 | 14 | 0.0000 | - | - |
| opus->gsm | 21 | 21 | 0.4286 | - | - |
| opus->mulaw | 16 | 16 | 0.1250 | - | - |
| opus->opus | 17 | 17 | 0.0000 | - | - |
| real->none | 156 | 0 | - | 0.0192 | 0.6417 |


### By device preset
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| bluetooth_mini | 25 | 25 | 0.2400 | - | - |
| budget_android | 12 | 12 | 0.2500 | - | - |
| car_speaker | 18 | 18 | 0.2778 | - | - |
| iphone_earpiece | 20 | 20 | 0.4000 | - | - |
| laptop_speaker | 19 | 19 | 0.2632 | - | - |
| pixel_speaker | 21 | 21 | 0.2857 | - | - |
| real_telephony | 156 | 0 | - | 0.0192 | 0.6417 |
| tablet | 23 | 23 | 0.3043 | - | - |
| watch_speaker | 18 | 18 | 0.1667 | - | - |


## Detector: `reverb`

| auc | eer | hit_at_fpr_1 | hit_at_fpr_2 | hit_at_fpr_5 |
|---|---|---|---|---|
| 0.8383 | 0.2628 | 0.4295 | 0.5256 | 0.5833 |

Operating threshold @ FPR 2%: `0.4625`

### By metadata label
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| real_direct | 156 | 0 | - | 0.0192 | 0.8383 |
| real_relay | 156 | 156 | 0.5256 | - | - |


### By codec2
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm | 56 | 56 | 0.5357 | - | - |
| mulaw | 52 | 52 | 0.5769 | - | - |
| none | 156 | 0 | - | 0.0192 | 0.8383 |
| opus | 48 | 48 | 0.4583 | - | - |


### By codec pair (codec1->codec2)
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm->gsm | 19 | 19 | 0.4211 | - | - |
| gsm->mulaw | 16 | 16 | 0.6875 | - | - |
| gsm->opus | 17 | 17 | 0.4706 | - | - |
| mulaw->gsm | 16 | 16 | 0.6250 | - | - |
| mulaw->mulaw | 20 | 20 | 0.6500 | - | - |
| mulaw->opus | 14 | 14 | 0.5000 | - | - |
| opus->gsm | 21 | 21 | 0.5714 | - | - |
| opus->mulaw | 16 | 16 | 0.3750 | - | - |
| opus->opus | 17 | 17 | 0.4118 | - | - |
| real->none | 156 | 0 | - | 0.0192 | 0.8383 |


### By device preset
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| bluetooth_mini | 25 | 25 | 0.4800 | - | - |
| budget_android | 12 | 12 | 0.6667 | - | - |
| car_speaker | 18 | 18 | 0.6111 | - | - |
| iphone_earpiece | 20 | 20 | 0.5500 | - | - |
| laptop_speaker | 19 | 19 | 0.5789 | - | - |
| pixel_speaker | 21 | 21 | 0.5238 | - | - |
| real_telephony | 156 | 0 | - | 0.0192 | 0.8383 |
| tablet | 23 | 23 | 0.3913 | - | - |
| watch_speaker | 18 | 18 | 0.5000 | - | - |


## Detector: `distortion`

| auc | eer | hit_at_fpr_1 | hit_at_fpr_2 | hit_at_fpr_5 |
|---|---|---|---|---|
| 0.9390 | 0.1314 | 0.7949 | 0.8013 | 0.8077 |

Operating threshold @ FPR 2%: `0.5961`

### By metadata label
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| real_direct | 156 | 0 | - | 0.0128 | 0.9390 |
| real_relay | 156 | 156 | 0.8013 | - | - |


### By codec2
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm | 56 | 56 | 0.7500 | - | - |
| mulaw | 52 | 52 | 0.8654 | - | - |
| none | 156 | 0 | - | 0.0128 | 0.9390 |
| opus | 48 | 48 | 0.7917 | - | - |


### By codec pair (codec1->codec2)
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm->gsm | 19 | 19 | 0.6842 | - | - |
| gsm->mulaw | 16 | 16 | 0.9375 | - | - |
| gsm->opus | 17 | 17 | 0.8824 | - | - |
| mulaw->gsm | 16 | 16 | 0.8125 | - | - |
| mulaw->mulaw | 20 | 20 | 0.8000 | - | - |
| mulaw->opus | 14 | 14 | 0.8571 | - | - |
| opus->gsm | 21 | 21 | 0.7619 | - | - |
| opus->mulaw | 16 | 16 | 0.8750 | - | - |
| opus->opus | 17 | 17 | 0.6471 | - | - |
| real->none | 156 | 0 | - | 0.0128 | 0.9390 |


### By device preset
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| bluetooth_mini | 25 | 25 | 0.8800 | - | - |
| budget_android | 12 | 12 | 1.0000 | - | - |
| car_speaker | 18 | 18 | 0.5000 | - | - |
| iphone_earpiece | 20 | 20 | 1.0000 | - | - |
| laptop_speaker | 19 | 19 | 0.4211 | - | - |
| pixel_speaker | 21 | 21 | 0.8571 | - | - |
| real_telephony | 156 | 0 | - | 0.0128 | 0.9390 |
| tablet | 23 | 23 | 0.7826 | - | - |
| watch_speaker | 18 | 18 | 1.0000 | - | - |


## Detector: `cnn_gbm_avg`

| auc | eer | hit_at_fpr_1 | hit_at_fpr_2 | hit_at_fpr_5 |
|---|---|---|---|---|
| 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

Operating threshold @ FPR 2%: `0.3654`

### By metadata label
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| real_direct | 156 | 0 | - | 0.0000 | 1.0000 |
| real_relay | 156 | 156 | 1.0000 | - | - |


### By codec2
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm | 56 | 56 | 1.0000 | - | - |
| mulaw | 52 | 52 | 1.0000 | - | - |
| none | 156 | 0 | - | 0.0000 | 1.0000 |
| opus | 48 | 48 | 1.0000 | - | - |


### By codec pair (codec1->codec2)
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| gsm->gsm | 19 | 19 | 1.0000 | - | - |
| gsm->mulaw | 16 | 16 | 1.0000 | - | - |
| gsm->opus | 17 | 17 | 1.0000 | - | - |
| mulaw->gsm | 16 | 16 | 1.0000 | - | - |
| mulaw->mulaw | 20 | 20 | 1.0000 | - | - |
| mulaw->opus | 14 | 14 | 1.0000 | - | - |
| opus->gsm | 21 | 21 | 1.0000 | - | - |
| opus->mulaw | 16 | 16 | 1.0000 | - | - |
| opus->opus | 17 | 17 | 1.0000 | - | - |
| real->none | 156 | 0 | - | 0.0000 | 1.0000 |


### By device preset
| slice | n | n_relay | recall | fpr | auc_vs_relay |
|---|---|---|---|---|---|
| bluetooth_mini | 25 | 25 | 1.0000 | - | - |
| budget_android | 12 | 12 | 1.0000 | - | - |
| car_speaker | 18 | 18 | 1.0000 | - | - |
| iphone_earpiece | 20 | 20 | 1.0000 | - | - |
| laptop_speaker | 19 | 19 | 1.0000 | - | - |
| pixel_speaker | 21 | 21 | 1.0000 | - | - |
| real_telephony | 156 | 0 | - | 0.0000 | 1.0000 |
| tablet | 23 | 23 | 1.0000 | - | - |
| watch_speaker | 18 | 18 | 1.0000 | - | - |

