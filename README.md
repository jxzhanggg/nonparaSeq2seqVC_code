# Non-para Seq2seq Voice Conversion

Implementation code of [Non-Parallel Sequence-to-Sequence Voice Conversion with Disentangled Linguistic and Speaker Representations](https://arxiv.org/abs/1906.10508).

For audio samples, please visit our [demo page](https://jxzhanggg.github.io/nonparaSeq2seqVC/).

![The structure overview of the model](struct.PNG)

## Dependencies

* Python 2.7
* PyTorch 1.0.0


## Usage

### Prepare training data
Prepare training data by downloading VCTK and CMU-ARCTIC datasets.
Extract the Mel-spectrograms and phonemes. Note that this repo
doesn't include this part code.
### Customize data reader
Modify the code of data reader to read your prepared training data.
```
.
├── pre-train
├───────────|── reader |
├───────────|──────────|── reader.py
├───────────|──────────|── symbols.py
.
├── fine-tune
├───────────|── reader |
├───────────|──────────|── reader.py
├───────────|──────────|── symbols.py

```
### Pre-train the model
Pre-train the model. 
```bash
$ cd pre-train
$ bash run.sh
```
Run the inference code to generate audio samples on multi-speaker dataset.
```bash
$ python inference.py
```
### Fine-tune the model
Fine-tune the model and generate audio samples on conversion pair. 
```bash
$ cd fine-tune
$ bash run.sh
```

## Reference
* "Non-Parallel Sequence-to-Sequence Voice Conversion with Disentangled Linguistic and Speaker Representations", Jing-Xuan Zhang, Zhen-Hua Ling, Li-Rong Dai, accepted by IEEE/ACM Transactions on Aduio, Speech and Language Processing, 2019.
* https://github.com/NVIDIA/tacotron2/