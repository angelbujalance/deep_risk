## Dual-Phase Cross-Modal Contrastive Learning for CMR-Guided ECG Representations for Cardiovascular Disease Assessment

<p align="center">
  <img src="https://github.com/angelbujalance/deep_risk/blob/main/Method-clip.png" width="480">
</p>

This is repository of the paper [Dual-Phase Cross-Modal Contrastive Learning for CMR-Guided ECG Representations for Cardiovascular Disease Assessment](https://spie.org/medical-imaging/presentation/Dual-phase-cross-modal-contrastive-learning-for-CMR-guided-ECG/13925-23). If you use this code, please cite:

```
@Article{DualPhaseCrossModalCLforECG2026,
  author = {Laura Alvarez Florez and Angel Bujalance Gomez and Femke Raijmakers and Samuel Ruiperez-Campillo and Maarten Kolk and Jesse Wiers and Julia Vogt and Erik Bekkers and Ivana Išgum and Fleur V. Y. Tjong},
  journal = {spie.org/medical-imaging/presentation/Dual-phase-cross-modal-contrastive-learning-for-CMR-guided-ECG/13925-23},
  title   = {Dual-Phase Cross-Modal Contrastive Learning for CMR-Guided ECG Representations for Cardiovascular Disease Assessment},
  year    = {2026},
}
```

## Instructions

### Masked data modelling - ECG Pretraining
Install [timm](https://github.com/oetu/pytorch-image-models/tree/3dbe2c484b7c5e44097427d5fcb50338df895b31/timm) library using `pip install -e pytorch-image-models`.
For detailed instructions to run the code for unimodal pre-training, see the [PRETRAIN.md](https://github.com/angelbujalance/mae/tree/ee025dd961584fa1cef5799162e9c2315a8b2755) of the mae forked repository. 

### CMR Pretraining
Install environment using `conda env create --cmr_pretrain/mae_ecg.yml`. 
To pretrain the CMR encoder, run the model with `ECG-CMR-CL/cmr_pretrain/run_CMR_pretrain.sh`.

### Multimodal contrastive learning
Once both models are pretrained, use the script `ECG-CMR-CL/cmr_pretrain/run_CMR_pretrain.sh` to perfrom the contrative learning between the ECG and CMR models run `ECG-CMR-CL/run_cl_triple.sh`.

### Fine-tuning / inference
Fine-tune the ECG model with `ECG-CMR-CL/fine_tune_ecg.sh`. 
