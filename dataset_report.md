# Dataset Report: FaceForensics++ Metadata

## Overview

This dataset contains video metadata for the FaceForensics++ collection and its manipulation variants. The workspace includes seven class folders and a set of CSV metadata files that describe the videos in each category.

The dataset is balanced at the CSV level: each class-specific file contains 1,000 records, and the combined metadata file contains 7,000 records total.

## Folder Structure

- `dataset/DeepFakeDetection/`
- `dataset/Deepfakes/`
- `dataset/Face2Face/`
- `dataset/FaceShifter/`
- `dataset/FaceSwap/`
- `dataset/NeuralTextures/`
- `dataset/original/`
- `dataset/csv/`

The `csv` folder contains per-class metadata files, a combined metadata file, a shuffled combined metadata file, and a summary statistics file.

## CSV Files

| File | Records | Label(s) | Notes |
| --- | ---: | --- | --- |
| `DeepFakeDetection.csv` | 1,000 | FAKE | Class-specific metadata |
| `Deepfakes.csv` | 1,000 | FAKE | Class-specific metadata |
| `Face2Face.csv` | 1,000 | FAKE | Class-specific metadata |
| `FaceShifter.csv` | 1,000 | FAKE | Class-specific metadata |
| `FaceSwap.csv` | 1,000 | FAKE | Class-specific metadata |
| `NeuralTextures.csv` | 1,000 | FAKE | Class-specific metadata |
| `original.csv` | 1,000 | REAL | Class-specific metadata |
| `FF++_Metadata.csv` | 7,000 | REAL / FAKE | Combined metadata |
| `FF++_Metadata_Shuffled.csv` | 7,000 | REAL / FAKE | Same combined metadata, shuffled |
| `Mean_Data.csv` | 7 | Class means | Per-class averages for frame count, resolution, and file size |

## Metadata Schema

The class CSVs share the same columns:

- Unnamed index column
- File Path
- Label
- Frame Count
- Width
- Height
- Codec
- File Size(MB)

The blank first column in the exported CSVs is an index column.

## Confirmed Dataset Properties

- All class CSVs use the `h264` codec.
- The `original.csv` file contains `REAL` examples.
- The manipulation categories contain `FAKE` examples.
- The combined metadata file includes 1,000 rows per class, for a total of 7,000 rows.
- The shuffled metadata file contains the same records as the combined file, but in randomized order.

## Per-Class Summary

| Class | Records | Label | Average Frame Count | Average Width | Average Height | Average File Size (MB) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| DeepFakeDetection | 1,000 | FAKE | 733.79 | 1,920.00 | 1,080.00 | 6.64 |
| Deepfakes | 1,000 | FAKE | 509.13 | 1,036.35 | 636.72 | 1.90 |
| Face2Face | 1,000 | FAKE | 509.13 | 1,030.91 | 636.72 | 1.86 |
| FaceShifter | 1,000 | FAKE | 509.13 | 1,036.35 | 636.72 | 1.83 |
| FaceSwap | 1,000 | FAKE | 406.14 | 1,036.35 | 636.72 | 1.56 |
| NeuralTextures | 1,000 | FAKE | 406.14 | 1,030.91 | 636.72 | 1.46 |
| original | 1,000 | REAL | 509.13 | 1,036.35 | 636.72 | 1.85 |

## Interpretation

The dataset is well suited for binary classification tasks such as real-versus-fake detection, as well as multi-class manipulation type classification. The balanced class distribution reduces bias toward any single category.

The summary statistics show that the classes differ in average duration, resolution, and file size, which may be useful for exploratory analysis or as auxiliary signals during modeling.

## Conclusion

This dataset provides a clean, balanced metadata collection for seven FaceForensics++ video categories. It includes both real and manipulated videos, together with consistent metadata fields that can support classification, analysis, and preprocessing workflows.