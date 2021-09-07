# Seq2Seq With Transformer

Implementation of [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf). This library is part of our
project: Building an AI library with ProtonX.

## Architecture Image

<p align="center">
    <img src="F:\8. BERTSum\BERTSum\assets\Transformer.png">
</p>

Authors:

- Github:
    - https://github.com/Xunino

Advisors:

- Github: https://github.com/bangoc123

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

## III. Training Process

Training script:

```bash
python train.py  --inp-lang=$inp_lang --tar-lang=$tar_lang \
                 --n_layers=2 --d-model=512 --diff-deep=2048 \
                 --min-sentence=0 --max-sentence=50 --bleu=True \
                 --warmup-steps=150 --epochs=1000
```

**Note**:

- If you want to retrain model, you can use this param: ```--retrain=True```
- Click [Here](https://colab.research.google.com/drive/11X9pk2rdBAjXVQugfqxPDezZCuj8_QD9#scrollTo=jqC_yVxZ4qje) to open
  notebook in google colab.

**There are some important arguments for the script you should consider when running it:**

- `dataset`: The folder of dataset
    - `train.en.txt`: input language
    - `train.vi.txt`: target language

## IV. Predict Process

```bash
python predict.py --test-path=${link_to_test_data} --inp-lang-path=${link_to_input_data} \
                  --tar-lang-path=${link_to_target_data} --hidden-units=128 \
                  --embedding-size=64 --min-sentence=10 --max-sentence=14 \
                  --train-mode="not_attention"
```

## V. Result and Comparision

```
-----------------------------------------------------------------
Input    :  <sos> they wrote almost a thousand pages on the topic <eos>
Predicted:  <sos> họ viết gần 1000 trang về của tranh của mình <eos> <eos> <eos>
Target   :  <sos> họ viết gần 1000 trang về chủ đề này <eos>
-----------------------------------------------------------------
Input    :  <sos> we blow it up and look at the pieces <eos>
Predicted:  <sos> chúng tôi cho nó nổ và xem xét từng mảnh nhỏ <eos> <eos>
Target   :  <sos> chúng tôi cho nó nổ và xem xét từng mảnh nhỏ <eos>
-----------------------------------------------------------------
Input    :  <sos> this is the euphore smog chamber in spain <eos>
Predicted:  <sos> đây là phòng nghiên cứu khói bụi euphore ở tây ban nha <eos>
Target   :  <sos> đây là phòng nghiên cứu khói bụi euphore ở tây ban nha <eos>
-----------------------------------------------------------------
Input    :  <sos> we also fly all over the world looking for this thing <eos>
Predicted:  <sos> chúng tôi còn bay khắp thế giới để tìm hiểu 50 tiết kiệm
Target   :  <sos> chúng tôi còn bay khắp thế giới để tìm phân tử này <eos>
-----------------------------------------------------------------
Input    :  <sos> this is the tower in the middle of the rainforest from above <eos>
Predicted:  <sos> đây chính là cái tháp giữa rừng sâu nhiều với 100 quốc <eos>
Target   :  <sos> đây chính là cái tháp giữa rừng sâu nhìn từ trên cao <eos>
-----------------------------------------------------------------
Input    :  <sos> christopher decharms a look inside the brain in real time <eos>
Predicted:  <sos> christopher decharms quét não bộ theo thời gian thực <eos> <eos> <eos> <eos>
Target   :  <sos> christopher decharms quét não bộ theo thời gian thực <eos>
=================================================================
Epoch 376 -- Loss: 30.585018157958984 -- Bleu_score: 0.6258203949505547
=================================================================
```

**Comments about these results:**
