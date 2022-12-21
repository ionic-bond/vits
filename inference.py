import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

import click
import torch
from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm



@click.group()
def cli():
    pass


@cli.command()
@click.option('--text', required=True)
@click.option('--model_path', required=True)
@click.option('--config_path', required=True)
@click.option('--output_path', required=True)
def run(text, model_path, config_path, output_path):
    hps = utils.get_hparams_from_file(config_path)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()

    _ = utils.load_checkpoint(model_path, net_g, None)

    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    write(output_path, hps.data.sampling_rate, audio)
