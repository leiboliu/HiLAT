# Hierarchical Label-wise Attention Transformer Model for Explainable ICD Coding

The source code is for the published research: [Hierarchical Label-wise Attention Transformer Model for Explainable ICD Coding](https://doi.org/10.1016/j.jbi.2022.104161)

## Prerequisites
Restore [MIMIC-III v1.4 data](https://physionet.org/content/mimiciii/1.4/) into a Postgres database. 

## Training data preparation
python3 hilat/data/mimic3_data_preparer.py \
    --data_output_dir=your_data_dir \
    --pre_process_level=level_1 \
    --segment_text=True 
    
## Training model
1. Use run_coding.sh to train the model on TPU environment
2. Train on GPU

python3 hilat/run_coding.py config.json

