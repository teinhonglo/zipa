import numpy as np
import scipy.stats as stats
from datasets import load_metric
from evaluate import load
from jiwer import compute_measures
import jiwer.transforms as tr
import time

def compute_metrics(total_metrics, pred_str, label_str):
    """ Computes metrics """

    #wer_metric = load("wer")
    #total_metrics['wer'] = wer_metric.compute(predictions=pred_str, references=label_str)
    total_metrics['wer'] = compute_wer(pred_str, label_str)
    pred_str_nosil = [sentence.replace("sil", "").strip() for sentence in pred_str]
    label_str_nosil = [sentence.replace("sil", "").strip() for sentence in label_str]
    total_metrics['wer_nosil'] = compute_wer(pred_str_nosil, label_str_nosil)
    #cer_metric = load("cer")
    #total_metrics['cer'] = cer_metric.compute(predictions=pred_str, references=label_str)
    total_metrics['cer'] = compute_cer(pred_str, label_str)

def compute_wer(predictions, references):
    #https://huggingface.co/spaces/evaluate-metric/wer/blob/1ff9fc0b838dea309eb52fb76f46b5b107286fc6/wer.py

    incorrect = 0
    total = 0
    for prediction, reference in zip(predictions, references):
        measures = compute_measures(reference, prediction)
        incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
        total += measures["substitutions"] + measures["deletions"] + measures["hits"]
    return incorrect / total


def compute_cer(predictions, references):
    # https://huggingface.co/spaces/evaluate-metric/cer/blob/af39b3b7914ffb126ece472884f75033cfc4727b/cer.py

    SENTENCE_DELIMITER = ""
    cer_transform = tr.Compose(
        [
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            tr.ReduceToSingleSentence(SENTENCE_DELIMITER),
            tr.ReduceToListOfListOfChars(),
        ]
    )
    incorrect = 0
    total = 0
    for prediction, reference in zip(predictions, references):
        measures = compute_measures(
            reference,
            prediction,
            truth_transform=cer_transform,
            hypothesis_transform=cer_transform,
        )
        incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
        total += measures["substitutions"] + measures["deletions"] + measures["hits"]
    return incorrect / total


if __name__ == '__main__':
    total_metrics = {}
    predictions = [
        "ih t w ah z b iy t ih ng ae n d v eh t ih ng ih n dh ah ae m b uh sh ah v d ao z b l ae p ih t s",
        "d eh n ae n d ae t s ah p ah hh iy t r ay d t uw f ae dh ah m hh ah"
    ]
    references = [
        "ih t w ah z b iy t ih ng ah n d w ey t ih ng ih n dh ah ae m b uh sh ah v dh ow z b l ae k sil p ih t s",
        "dh eh n sil ae n d ae t s ah p er hh iy t r ay d t uw f ae dh ah m hh er"
    ]

    # faster
    st = time.time()
    print("jiwer wer: {}".format(compute_wer(predictions, references)))
    print("jiwer cer: {}".format(compute_cer(predictions, references)))
    ed = time.time()
    print("jiwer time: {}".format(ed-st))

    # slower
    st = time.time()
    wer_metric = load("wer")
    print('huggingface wer: {}'.format(wer_metric.compute(predictions=predictions, references=references)))
    cer_metric = load("cer")
    print('huggingface cer: {}'.format(cer_metric.compute(predictions=predictions, references=references)))
    ed = time.time()
    print("huggingface time: {}".format(ed-st))