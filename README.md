# Seq2Seq With Transformer

Design Machine Translation Engine for Vietnamese using Transformer Architecture from
paper [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf). Give us a star if you like this repo.

## Architecture Image

<p align="center">
    <img src="https://github.com/Xunino/Seq2SeqWithTransformer/blob/main/assets/Transformer.png">
</p>

Authors:

- Github: Xunino
- Email: ndlinh.ai@gmail.com

## I. Set up environment

- Step 1:

```bash
conda create -n {your_env_name} python==3.7.0
```

- Step 2:

```bash
conda env create -f environment.yml
```

- Step 3:

```bash
conda activate {your_env_name}
``` 

## II. Set up your dataset

- Guide user how to download your data and set the data pipeline

- References: [NLP](https://github.com/Xunino/Seq2SeqWithTransformer/tree/main/dataset/seq2seq)

- Data pipeline example:

| train.en   |   train.vi      |
|----------|:-------------:|
| I love you       |  Tôi yêu bạn|
| ....             |    .... |

**Note:** You can use your data for train to this model.

## III. Training Process

Training script:

```bash
python train.py  --input-path=${path_to_test_data} --target-path=${path_to_input_data} --n_layers=2 --header-size=8 --d-model=512 --diff-deep=2048 --min-sentence=0 --max-sentence=50 --bleu=True
```

**There are some important arguments for the script you should consider when running it:**

- `dataset`: The folder of dataset
    - `train.en.txt`: input language
    - `train.vi.txt`: target language
- `--input-path`: input language path (E.g. /dataset/seq2seq/train.en.txt)
- `--target-path`: target language path (E.g. /dataset/seq2seq/train.vi.txt)
- `--n_layers`: Number Encode vs Decode layers, default is 2
- `--header-size`: Number Multi Head Attention, default is 8
- `--d-model`: The dimension of linear projection for all sentence. It was mentioned in Section `3.2.2` on
  the [page 5](https://arxiv.org/pdf/1706.03762.pdf)
- `--diff-deep`: Hidden size in Position-Wise Feed-Forward Network. It was mentioned in Section `3.3`
- `--min-sentence`: Min length of sentence to filter in dataset
- `--max-sentence`: Max length of sentence to filter in dataset
- `--bleu`: bool values. It's using to evaluate NLP model ([More](https://aclanthology.org/P02-1040.pdf))

**Note**:

- If you want to retrain model, you can use this param: ```--retrain=True```
- Click [Here](https://colab.research.google.com/drive/1mxS6_1QzGMPuGSNAg5N-FjKjflneZgbY?usp=sharing) to open notebook
  in google colab.

## IV. Predict Process

```bash
python translation.py --input-path=${path_to_test_data} --target-path=${path_to_input_data}
```

## V. Result

```
===========================================================
Epoch 56 -- Batch: 0 -- Loss: 0.0391 -- Accuracy: 0.9669
Epoch 56 -- Batch: 50 -- Loss: 0.0359 -- Accuracy: 0.9726
Epoch 56 -- Batch: 100 -- Loss: 0.0349 -- Accuracy: 0.9733
Epoch 56 -- Batch: 150 -- Loss: 0.0343 -- Accuracy: 0.9736
Epoch 56 -- Batch: 200 -- Loss: 0.0338 -- Accuracy: 0.9739
Epoch 56 -- Batch: 250 -- Loss: 0.0337 -- Accuracy: 0.9739
Epoch 56 -- Batch: 300 -- Loss: 0.0336 -- Accuracy: 0.9741
Epoch 56 -- Batch: 350 -- Loss: 0.0336 -- Accuracy: 0.9740
Epoch 56 -- Batch: 400 -- Loss: 0.0335 -- Accuracy: 0.9741
Epoch 56 -- Batch: 450 -- Loss: 0.0335 -- Accuracy: 0.9742
Epoch 56 -- Batch: 500 -- Loss: 0.0335 -- Accuracy: 0.9742
Epoch 56 -- Batch: 550 -- Loss: 0.0333 -- Accuracy: 0.9743
Epoch 56 -- Batch: 600 -- Loss: 0.0334 -- Accuracy: 0.9743
Epoch 56 -- Batch: 650 -- Loss: 0.0334 -- Accuracy: 0.9743
Epoch 56 -- Batch: 700 -- Loss: 0.0335 -- Accuracy: 0.9742
Epoch 56 -- Batch: 750 -- Loss: 0.0336 -- Accuracy: 0.9742
Epoch 56 -- Batch: 800 -- Loss: 0.0335 -- Accuracy: 0.9743
Epoch 56 -- Batch: 850 -- Loss: 0.0335 -- Accuracy: 0.9742
Epoch 56 -- Batch: 900 -- Loss: 0.0334 -- Accuracy: 0.9742
Epoch 56 -- Batch: 950 -- Loss: 0.0335 -- Accuracy: 0.9742
-----------------------------------------------------------
Epoch 56 -- Loss: 0.0335 -- Accuracy: 0.9742 
===========================================================
```

**Comments about these results:**
