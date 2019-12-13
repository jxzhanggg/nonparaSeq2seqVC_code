# Non-parallel Seq2seq Voice Conversion

Implementation code of [Non-Parallel Sequence-to-Sequence Voice Conversion with Disentangled Linguistic and Speaker Representations](https://arxiv.org/abs/1906.10508).

For audio samples, please visit our [demo page](https://jxzhanggg.github.io/nonparaSeq2seqVC/).

![The structure overview of the model](struct.PNG)

## Dependencies

* Python 2.7
* PyTorch 1.0.1
* CUDA 10.0

## Usage

### Prepare training data
Prepare training data by downloading [VCTK](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) and [CMU-ARCTIC](http://www.speech.cs.cmu.edu/cmu_arctic/packed/) datasets.
Extract the Mel-spectrograms and phonemes. This repo
doesn't include this part code. However, you can refer to 
the preprocess code of [Deepvoice3_pytorch](https://github.com/r9y9/deepvoice3_pytorch).
### Customize data reader
Modify the following code of data reader to read your prepared training data.
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
* "Sequence-to-Sequence Acoustic Modeling for Voice Conversion", Jing-Xuan Zhang, Zhen-Hua Ling, Li-Juan Liu, Yuan Jiang, Li-Rong Dai, IEEE/ACM Transactions on Audio, Speech and Language Processing, vol. 27, no. 3, pp. 631-644, March 2019.
* "Forward Attention in Sequence-to-sequence Acoustic Modelling for Speech Synthesis", Jing-Xuan Zhang, Zhen-Hua Ling, Li-Rong Dai, ICASSP, pp. 4789–4793, 2018.

## Acknowledgements
Part of code was adapted from the following project:
* https://github.com/NVIDIA/tacotron2/