# Model limits and bias

**This software is for education only and is not for clinical use.** It is not a medical device, has not been
validated on any clinical population, and must not inform any diagnosis or treatment decision.

## Chest X-ray classifier (DenseNet121 on NIH ChestX-ray14)
- **Noisy labels.** ChestX-ray14 labels were extracted from radiology reports by text mining, not read from the images by experts. Some labels are wrong, and the model learns those errors.
- **Single source.** All images come from one hospital system. Performance on other scanners, protocols, hospitals or countries is unknown and is likely lower. No external validation was done.
- **Population bias.** The dataset reflects one patient population. Differences in age, sex, race and body habitus between that population and a deployment population can change error rates, and this project did not measure subgroup performance.
- **Shortcut learning.** Chest X-ray models are known to pick up non-pathological cues such as tubes, markers, portable vs. fixed equipment and image burn-in. A high AUC does not prove the model looks at disease.
- **Weak classes.** Rare findings (for example Hernia) have few positives. Their AUCs are unstable and the model should not be trusted for them. Check per-class numbers in `reports/classifier_metrics.md`.
- **Confidence is not calibrated.** Training used a plain BCE loss with no calibration step. A 0.40 output does not mean a 40% chance of disease.
- **Grad-CAM is a coarse explanation.** It shows which regions raised a class score, at low resolution. It can highlight the wrong place on a correct prediction, and it is not a lesion segmentation.
- **Not for other body parts or modalities.** The classifier is meant for frontal chest radiographs only.

## CT segmentation (MONAI 3D U-Net on MSD Task09 Spleen)
- **Tiny training set.** The task provides 41 labelled CT volumes from a single institution. With no separate test set, the reported Dice is measured on the validation split that also chose the checkpoint, so it is optimistic.
- **Narrow scope.** It segments the spleen on portal-venous CT only. Other organs, phases, contrast protocols, pediatric patients or heavily abnormal anatomy are out of distribution.
- **Orientation assumption.** The serving code assumes axial slices in standard DICOM patient orientation. Oblique or reformatted series can give wrong masks.
- **Requires at least 20 slices.** Shorter series are rejected. MR is rejected because the model never saw MR.

## System limits
- The viewer identifies reviewers by a typed name. There is no authentication or authorization. A production system needs both.
- The audit log is tamper-evident (hash chain) but lives in the same database as the data it protects. It is not a legally sufficient record.
- No de-identification step. Use only public, already de-identified data or synthetic data.
- Synthetic DICOM from `app.sample` is for testing the plumbing and says nothing about model quality.
- `DEMO_MODE=true` returns placeholder outputs from an untrained stub. Never report or screenshot it as real results.
