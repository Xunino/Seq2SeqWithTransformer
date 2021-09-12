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
Epoch 100 -- Batch: 0 -- Loss: 0.0359 -- Accuracy: 0.9779
Epoch 100 -- Batch: 50 -- Loss: 0.0312 -- Accuracy: 0.9761
Epoch 100 -- Batch: 100 -- Loss: 0.0312 -- Accuracy: 0.9761
Epoch 100 -- Batch: 150 -- Loss: 0.0315 -- Accuracy: 0.9758
Epoch 100 -- Batch: 200 -- Loss: 0.0313 -- Accuracy: 0.9759
Epoch 100 -- Batch: 250 -- Loss: 0.0313 -- Accuracy: 0.9759
Epoch 100 -- Batch: 300 -- Loss: 0.0312 -- Accuracy: 0.9760
Epoch 100 -- Batch: 350 -- Loss: 0.0312 -- Accuracy: 0.9760
Epoch 100 -- Batch: 400 -- Loss: 0.0312 -- Accuracy: 0.9760
Epoch 100 -- Batch: 450 -- Loss: 0.0312 -- Accuracy: 0.9760
Epoch 100 -- Batch: 500 -- Loss: 0.0312 -- Accuracy: 0.9760
Epoch 100 -- Batch: 550 -- Loss: 0.0312 -- Accuracy: 0.9759
Epoch 100 -- Batch: 600 -- Loss: 0.0313 -- Accuracy: 0.9759
Epoch 100 -- Batch: 650 -- Loss: 0.0314 -- Accuracy: 0.9759
Epoch 100 -- Batch: 700 -- Loss: 0.0312 -- Accuracy: 0.9760
Epoch 100 -- Batch: 750 -- Loss: 0.0313 -- Accuracy: 0.9760
Epoch 100 -- Batch: 800 -- Loss: 0.0312 -- Accuracy: 0.9760
Epoch 100 -- Batch: 850 -- Loss: 0.0312 -- Accuracy: 0.9760
Epoch 100 -- Batch: 900 -- Loss: 0.0311 -- Accuracy: 0.9760
Epoch 100 -- Batch: 950 -- Loss: 0.0312 -- Accuracy: 0.9760
-----------------------------------------------------------
Input   :  <sos> and the speed with which they associate it with quot slumdog millionaire quot or the favelas in rio speaks to the enduring nature <eos>
Predict :  <sos> và tốc độ mà chúng liên hệ nó với quot triệu phú khu ổ chuột quot hay những khu phố quot favela quot ở rio nói lên bản chất bền vững đó <eos>
Target  :  <sos> và tốc độ mà chúng liên hệ nó với quot triệu phú khu ổ chuột quot hay những khu phố quot favela quot ở rio nói lên bản chất bền vững đó <eos>
-----------------------------------------------------------
Input   :  <sos> in a filmclub season about democracy and government we screened quot mr smith goes to washington quot <eos>
Predict :  <sos> trong mùa chiếu của câu lạc bộ phim về dân chủ và chính quyền chúng tôi đã chiếu quot ông smith đến washington quot <eos>
Target  :  <sos> trong mùa chiếu của câu lạc bộ phim về dân chủ và chính quyền chúng tôi đã chiếu quot ông smith đến washington quot <eos>
-----------------------------------------------------------
Input   :  <sos> made in 1939 the film is older than most of our members apos grandparents <eos>
Predict :  <sos> được làm vào năm 1939 bộ phim có tuổi già hơn tuổi của hầu hết ông bà của các thành viên <eos>
Target  :  <sos> được làm vào năm 1939 bộ phim có tuổi già hơn tuổi của hầu hết ông bà của các thành viên <eos>
-----------------------------------------------------------
Input   :  <sos> frank capra apos s classic values independence and propriety <eos>
Predict :  <sos> sự cổ điển của frank capra có giá trị ở tính độc lập và sự thích nghi <eos>
Target  :  <sos> sự cổ điển của frank capra có giá trị ở tính độc lập và sự thích nghi <eos>
-----------------------------------------------------------
Input   :  <sos> it shows how to do right how to be heroically awkward <eos>
Predict :  <sos> bộ phim chỉ ra làm thế nào để làm đúng làm thế nào để trở nên kì lạ phi thường <eos>
Target  :  <sos> bộ phim chỉ ra làm thế nào để làm đúng làm thế nào để trở nên kì lạ phi thường <eos>
-----------------------------------------------------------
Epoch 100 -- Loss: 0.0312 -- Accuracy: 0.9760 -- Bleu_score: 0.9740
===========================================================
```