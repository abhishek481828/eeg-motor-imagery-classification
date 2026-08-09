# Dataset Instructions: PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0

## Manual Download Instructions

The dataset must be manually downloaded by the user and placed in the designated raw data folder. **Do not commit raw dataset files to Git.**

### 1. Download Location
Download the dataset from PhysioNet:
- **URL**: https://physionet.org/content/eegmmidb/1.0.0/
- **CLI Download** (using `wget` or `curl`):
  ```bash
  wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/ -P data/raw/physionet/
  ```

### 2. Required Directory Structure
Extract/place the downloaded files into:
```
data/
└── raw/
    └── physionet/
        ├── S001/
        │   ├── S001R01.edf
        │   ├── S001R02.edf
        │   └── ...
        ├── S002/
        ├── ...
        └── S109/
```

### 3. Verification
Run the dataset inspection script to check data integrity:
```bash
make inspect-data
# OR
python scripts/inspect_dataset.py
```
