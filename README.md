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

- `./dataset/seq2seq/`: The folder of dataset
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

- If you want to retrain model, you can use this param: `--retrain=True`
- <a href="https://colab.research.google.com/drive/1mxS6_1QzGMPuGSNAg5N-FjKjflneZgbY?usp=sharing" target="_blank">
  <img src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667">

</a>

## IV. Predict Process

```bash
python translation.py --n_layers=2 --header-size=8 --d-model=512 --diff-deep=2048 --max-sentence=50
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

Other result

```
===========================================================
Epoch 80 -- Batch: 0 -- Loss: 0.0795 -- Accuracy: 0.9415
Epoch 80 -- Batch: 50 -- Loss: 0.0871 -- Accuracy: 0.9354
Epoch 80 -- Batch: 100 -- Loss: 0.0884 -- Accuracy: 0.9350
Epoch 80 -- Batch: 150 -- Loss: 0.0891 -- Accuracy: 0.9343
Epoch 80 -- Batch: 200 -- Loss: 0.0891 -- Accuracy: 0.9342
Epoch 80 -- Batch: 250 -- Loss: 0.0891 -- Accuracy: 0.9340
Epoch 80 -- Batch: 300 -- Loss: 0.0892 -- Accuracy: 0.9338
Epoch 80 -- Batch: 350 -- Loss: 0.0894 -- Accuracy: 0.9339
Epoch 80 -- Batch: 400 -- Loss: 0.0895 -- Accuracy: 0.9338
Epoch 80 -- Batch: 450 -- Loss: 0.0899 -- Accuracy: 0.9335
Epoch 80 -- Batch: 500 -- Loss: 0.0902 -- Accuracy: 0.9333
Epoch 80 -- Batch: 550 -- Loss: 0.0906 -- Accuracy: 0.9330
Epoch 80 -- Batch: 600 -- Loss: 0.0910 -- Accuracy: 0.9327
Epoch 80 -- Batch: 650 -- Loss: 0.0911 -- Accuracy: 0.9326
Epoch 80 -- Batch: 700 -- Loss: 0.0915 -- Accuracy: 0.9323
Epoch 80 -- Batch: 750 -- Loss: 0.0918 -- Accuracy: 0.9321
Epoch 80 -- Batch: 800 -- Loss: 0.0922 -- Accuracy: 0.9318
Epoch 80 -- Batch: 850 -- Loss: 0.0926 -- Accuracy: 0.9316
Epoch 80 -- Batch: 900 -- Loss: 0.0928 -- Accuracy: 0.9314
Epoch 80 -- Batch: 950 -- Loss: 0.0931 -- Accuracy: 0.9313
-----------------------------------------------------------
Input   :  <sos> good then i shall attempt the impossible or at least the improbable <eos>
Predict :  <sos> tốt vậy là tôi sẽ cố gắng làm điều không tưởng <eos>
Target  :  <sos> tốt vậy tôi sẽ cố gắng làm điều bất khả thi này hoặc ít nhất là điều khó xảy ra <eos>
-----------------------------------------------------------
Input   :  <sos> that doesn apos t mean you have to go to an mba program <eos>
Predict :  <sos> điều đó không có nghĩa bạn phải theo học chương trình mba <eos>
Target  :  <sos> điều đó không có nghĩa bạn phải theo học chương trình mba <eos>
-----------------------------------------------------------
Input   :  <sos> and you can match that demand hour by hour for the whole year almost <eos>
Predict :  <sos> và bạn có thể khớp nhu cầu đó từng giờ cho hầu như cả năm <eos>
Target  :  <sos> và bạn có thể khớp nhu cầu đó từng giờ cho hầu như cả năm <eos>
-----------------------------------------------------------
Input   :  <sos> i struggled to 91 unclear 93 day was a huge issue <eos>
Predict :  <sos> tôi vật lộn kiếm sống qua ngày 91 không rõ 93 là một vấn đề lớn <eos>
Target  :  <sos> tôi vật lộn kiếm sống qua ngày 91 không rõ 93 là một vấn đề lớn <eos>
-----------------------------------------------------------
Input   :  <sos> i know i should mention i apos m making all these things <eos>
Predict :  <sos> tôi biết là mình nên lưu ý là tôi làm ra tất cả những thứ này <eos>
Target  :  <sos> tôi biết là mình nên lưu ý là tôi làm ra tất cả những thứ này <eos>
-----------------------------------------------------------
Epoch 80 -- Loss: 0.0931 -- Accuracy: 0.9313 -- Bleu_score: 0.8431
===========================================================
```