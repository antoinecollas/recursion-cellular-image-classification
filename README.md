# Recursion-cellular-image-classification
Code to finish 42/866.

https://www.kaggle.com/c/recursion-cellular-image-classification/

# Instructions

## Installation

Create environment:

```
conda create -n recursion-cellular-image-classification python=3.7.3
conda activate recursion-cellular-image-classification
```

Install packages:

```
pip install -r requirement.txt
```

## Data
Download data from [Kaggle](https://www.kaggle.com/c/recursion-cellular-image-classification/data).

Unzip train.zip and test.zip.

Repertory should look like this:
```
data/
    metadata/
        train.csv
        test.csv
        train_controls.csv
        test_controls.csv
    train/
    test/
```

Convert data from png to jpeg:

```
python png_to_jpeg.py
```

## Training and test
```
python compute_stats_experiments.py
python main.py
```