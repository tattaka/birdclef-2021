# BirdCLEF 2021 - Birdcall Identification 4th place solution

## My solution detail
[kaggle discussion](https://www.kaggle.com/c/birdclef-2021/discussion/243293)  
[Inference Notebook (best submission)](https://www.kaggle.com/tattaka/birdclef2021-submissions-pp-ave?scriptVersionId=64016465)

## Environment  
Use [Kaggle Docker](https://gcr.io/kaggle-gpu-images/python).
Then install the missing packages according to the `requiement.txt`.

## Usage
1. Prepare `train_metadata.csv`.   
Execute all cells `src/prepare_train_df.ipynb`.  
2. Execute in the order of `src/stage1/train.sh`, `src/stage2/train.sh`, `src/finetune_30/train.sh`.

## License
MIT