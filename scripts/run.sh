DATASET=$1

# SASRec
python run.py -m seed=42,221,451,934,1984 dataset=$DATASET

# SASRec with content initialization
python run.py -m seed=42,221,451,934,1984 dataset=$DATASET use_pretrained_item_embeddings=True

# SASRec with trainable delta
python run.py -m seed=42,221,451,934,1984 dataset=$DATASET use_pretrained_item_embeddings=True train_delta=True