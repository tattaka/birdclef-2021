# stage1: fix save directry, 4fold for pseudo labeling
import argparse
import os

# import random
import warnings

# from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy as sp
import soundfile as sf
import timm
import torch

# from pytorch_lightning.utilities import rank_zero_info
from sklearn import model_selection
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from tqdm import tqdm, trange

# import glob


warnings.simplefilter("ignore")


train_df = pd.read_csv("../../input/birdclef-2021/train_metadata.csv")
train_soundscape_labels = pd.read_csv(
    "../../input/birdclef-2021/train_soundscape_labels.csv"
)
valid_data = Path("../../input/birdclef-2021/train_soundscapes/")
valid_all_audios = list(valid_data.glob("*.ogg"))
target_columns = [
    "acafly",
    "acowoo",
    "aldfly",
    "ameavo",
    "amecro",
    "amegfi",
    "amekes",
    "amepip",
    "amered",
    "amerob",
    "amewig",
    "amtspa",
    "andsol1",
    "annhum",
    "astfly",
    "azaspi1",
    "babwar",
    "baleag",
    "balori",
    "banana",
    "banswa",
    "banwre1",
    "barant1",
    "barswa",
    "batpig1",
    "bawswa1",
    "bawwar",
    "baywre1",
    "bbwduc",
    "bcnher",
    "belkin1",
    "belvir",
    "bewwre",
    "bkbmag1",
    "bkbplo",
    "bkbwar",
    "bkcchi",
    "bkhgro",
    "bkmtou1",
    "bknsti",
    "blbgra1",
    "blbthr1",
    "blcjay1",
    "blctan1",
    "blhpar1",
    "blkpho",
    "blsspa1",
    "blugrb1",
    "blujay",
    "bncfly",
    "bnhcow",
    "bobfly1",
    "bongul",
    "botgra",
    "brbmot1",
    "brbsol1",
    "brcvir1",
    "brebla",
    "brncre",
    "brnjay",
    "brnthr",
    "brratt1",
    "brwhaw",
    "brwpar1",
    "btbwar",
    "btnwar",
    "btywar",
    "bucmot2",
    "buggna",
    "bugtan",
    "buhvir",
    "bulori",
    "burwar1",
    "bushti",
    "butsal1",
    "buwtea",
    "cacgoo1",
    "cacwre",
    "calqua",
    "caltow",
    "cangoo",
    "canwar",
    "carchi",
    "carwre",
    "casfin",
    "caskin",
    "caster1",
    "casvir",
    "categr",
    "ccbfin",
    "cedwax",
    "chbant1",
    "chbchi",
    "chbwre1",
    "chcant2",
    "chispa",
    "chswar",
    "cinfly2",
    "clanut",
    "clcrob",
    "cliswa",
    "cobtan1",
    "cocwoo1",
    "cogdov",
    "colcha1",
    "coltro1",
    "comgol",
    "comgra",
    "comloo",
    "commer",
    "compau",
    "compot1",
    "comrav",
    "comyel",
    "coohaw",
    "cotfly1",
    "cowscj1",
    "cregua1",
    "creoro1",
    "crfpar",
    "cubthr",
    "daejun",
    "dowwoo",
    "ducfly",
    "dusfly",
    "easblu",
    "easkin",
    "easmea",
    "easpho",
    "eastow",
    "eawpew",
    "eletro",
    "eucdov",
    "eursta",
    "fepowl",
    "fiespa",
    "flrtan1",
    "foxspa",
    "gadwal",
    "gamqua",
    "gartro1",
    "gbbgul",
    "gbwwre1",
    "gcrwar",
    "gilwoo",
    "gnttow",
    "gnwtea",
    "gocfly1",
    "gockin",
    "gocspa",
    "goftyr1",
    "gohque1",
    "goowoo1",
    "grasal1",
    "grbani",
    "grbher3",
    "grcfly",
    "greegr",
    "grekis",
    "grepew",
    "grethr1",
    "gretin1",
    "greyel",
    "grhcha1",
    "grhowl",
    "grnher",
    "grnjay",
    "grtgra",
    "grycat",
    "gryhaw2",
    "gwfgoo",
    "haiwoo",
    "heptan",
    "hergul",
    "herthr",
    "herwar",
    "higmot1",
    "hofwoo1",
    "houfin",
    "houspa",
    "houwre",
    "hutvir",
    "incdov",
    "indbun",
    "kebtou1",
    "killde",
    "labwoo",
    "larspa",
    "laufal1",
    "laugul",
    "lazbun",
    "leafly",
    "leasan",
    "lesgol",
    "lesgre1",
    "lesvio1",
    "linspa",
    "linwoo1",
    "littin1",
    "lobdow",
    "lobgna5",
    "logshr",
    "lotduc",
    "lotman1",
    "lucwar",
    "macwar",
    "magwar",
    "mallar3",
    "marwre",
    "mastro1",
    "meapar",
    "melbla1",
    "monoro1",
    "mouchi",
    "moudov",
    "mouela1",
    "mouqua",
    "mouwar",
    "mutswa",
    "naswar",
    "norcar",
    "norfli",
    "normoc",
    "norpar",
    "norsho",
    "norwat",
    "nrwswa",
    "nutwoo",
    "oaktit",
    "obnthr1",
    "ocbfly1",
    "oliwoo1",
    "olsfly",
    "orbeup1",
    "orbspa1",
    "orcpar",
    "orcwar",
    "orfpar",
    "osprey",
    "ovenbi1",
    "pabspi1",
    "paltan1",
    "palwar",
    "pasfly",
    "pavpig2",
    "phivir",
    "pibgre",
    "pilwoo",
    "pinsis",
    "pirfly1",
    "plawre1",
    "plaxen1",
    "plsvir",
    "plupig2",
    "prowar",
    "purfin",
    "purgal2",
    "putfru1",
    "pygnut",
    "rawwre1",
    "rcatan1",
    "rebnut",
    "rebsap",
    "rebwoo",
    "redcro",
    "reevir1",
    "rehbar1",
    "relpar",
    "reshaw",
    "rethaw",
    "rewbla",
    "ribgul",
    "rinkin1",
    "roahaw",
    "robgro",
    "rocpig",
    "rotbec",
    "royter1",
    "rthhum",
    "rtlhum",
    "ruboro1",
    "rubpep1",
    "rubrob",
    "rubwre1",
    "ruckin",
    "rucspa1",
    "rucwar",
    "rucwar1",
    "rudpig",
    "rudtur",
    "rufhum",
    "rugdov",
    "rumfly1",
    "runwre1",
    "rutjac1",
    "saffin",
    "sancra",
    "sander",
    "savspa",
    "saypho",
    "scamac1",
    "scatan",
    "scbwre1",
    "scptyr1",
    "scrtan1",
    "semplo",
    "shicow",
    "sibtan2",
    "sinwre1",
    "sltred",
    "smbani",
    "snogoo",
    "sobtyr1",
    "socfly1",
    "solsan",
    "sonspa",
    "soulap1",
    "sposan",
    "spotow",
    "spvear1",
    "squcuc1",
    "stbori",
    "stejay",
    "sthant1",
    "sthwoo1",
    "strcuc1",
    "strfly1",
    "strsal1",
    "stvhum2",
    "subfly",
    "sumtan",
    "swaspa",
    "swathr",
    "tenwar",
    "thbeup1",
    "thbkin",
    "thswar1",
    "towsol",
    "treswa",
    "trogna1",
    "trokin",
    "tromoc",
    "tropar",
    "tropew1",
    "tuftit",
    "tunswa",
    "veery",
    "verdin",
    "vigswa",
    "warvir",
    "wbwwre1",
    "webwoo1",
    "wegspa1",
    "wesant1",
    "wesblu",
    "weskin",
    "wesmea",
    "westan",
    "wewpew",
    "whbman1",
    "whbnut",
    "whcpar",
    "whcsee1",
    "whcspa",
    "whevir",
    "whfpar1",
    "whimbr",
    "whiwre1",
    "whtdov",
    "whtspa",
    "whwbec1",
    "whwdov",
    "wilfly",
    "willet1",
    "wilsni1",
    "wiltur",
    "wlswar",
    "wooduc",
    "woothr",
    "wrenti",
    "y00475",
    "yebcha",
    "yebela1",
    "yebfly",
    "yebori1",
    "yebsap",
    "yebsee1",
    "yefgra1",
    "yegvir",
    "yehbla",
    "yehcar1",
    "yelgro",
    "yelwar",
    "yeofly1",
    "yerwar",
    "yeteup1",
    "yetvir",
]
bird2id = {b: i for i, b in enumerate(target_columns)}
id2bird = {i: b for i, b in enumerate(target_columns)}
split = "StratifiedKFold"
split_params = {"n_splits": 4, "shuffle": True, "random_state": 1213}
splitter = getattr(model_selection, split)(**split_params)


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, sr):
        for trns in self.transforms:
            y = trns(y, sr)
        return y


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray, sr):
        if self.always_apply:
            return self.apply(y, sr=sr)
        else:
            if np.random.rand() < self.p:
                return self.apply(y, sr=sr)
            else:
                return y

    def apply(self, y: np.ndarray, **params):
        raise NotImplementedError


class Normalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        return np.asfortranarray(y_vol)


class BirdClef2021Dataset(Dataset):
    def __init__(
        self,
        clip,
    ):
        self.clip = clip
        self.wave_transforms = Compose(
            [
                Normalize(p=1),
            ]
        )

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        waveform = self.wave_transforms(self.clip, sr=32000)
        waveform = torch.Tensor(np.nan_to_num(waveform))
        return {
            "wave": waveform,
        }


class SoundscapeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        clip: np.ndarray,
        train_period=30,
    ):
        self.df = df
        self.clip = np.concatenate([clip, clip, clip])
        self.train_period = train_period
        self.waveform_transforms = Compose(
            [
                Normalize(p=1),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        SR = 32000
        sample = self.df.loc[idx, :]
        row_id = sample.row_id
        birds = (
            train_soundscape_labels[train_soundscape_labels["row_id"] == row_id][
                "birds"
            ]
            .map(lambda s: s.split(" "))
            .values[0]
        )
        end_seconds = int(sample.seconds)
        start_seconds = int(end_seconds - 5)

        end_index = int(
            SR * (end_seconds + (self.train_period - 5) / 2) + len(self.clip) // 3
        )
        start_index = int(
            SR * (start_seconds - (self.train_period - 5) / 2) + len(self.clip) // 3
        )

        y = self.clip[start_index:end_index].astype(np.float32)

        y = np.nan_to_num(y)

        if self.waveform_transforms:
            y = self.waveform_transforms(y, SR)

        y = np.nan_to_num(y)

        target = np.zeros(397, dtype=np.float32)
        for b in birds:
            if b == "nocall":
                continue
            else:
                target[bird2id[b]] = 1.0
        target = torch.Tensor(target)
        return y, row_id, target


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(nn.Module):
    """
    Simple class for flattening layer.
    """

    def forward(self, x):
        return x.view(x.size()[0], -1)


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)


def gem_freq(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1.0 / p)


class GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem_freq(x, p=self.p, eps=self.eps)


class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        norm_min, norm_max = Xstd.min(-1)[0].min(-1)[0], Xstd.max(-1)[0].max(-1)[0]
        fix_ind = (norm_max - norm_min) > self.eps * torch.ones_like(
            (norm_max - norm_min)
        )
        V = torch.zeros_like(Xstd)
        if fix_ind.sum():
            V_fix = Xstd[fix_ind]
            norm_max_fix = norm_max[fix_ind, None, None]
            norm_min_fix = norm_min[fix_ind, None, None]
            V_fix = torch.max(
                torch.min(V_fix, norm_max_fix),
                norm_min_fix,
            )
            # print(V_fix.shape, norm_min_fix.shape, norm_max_fix.shape)
            V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
            V[fix_ind] = V_fix
        return V


class AttHead(nn.Module):
    def __init__(
        self, in_chans, p=0.5, num_class=397, train_period=15.0, infer_period=5.0
    ):
        super().__init__()
        self.train_period = train_period
        self.infer_period = infer_period
        self.pooling = GeMFreq()

        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),
            nn.Linear(in_chans, 512),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.attention = nn.Conv1d(
            in_channels=512,
            out_channels=num_class,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.fix_scale = nn.Conv1d(
            in_channels=512,
            out_channels=num_class,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, feat):
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)  # (bs, time, ch)

        feat = self.dense_layers(feat).permute(0, 2, 1)  # (bs, 512, time)
        time_att = torch.tanh(self.attention(feat))
        # TODO: ここで時間軸でattention取ってるけどchannel軸でattention取っても面白そう(multi labelなので考えないといけない)
        # attentionについても考えないといけない(本来はout * attではなくout + out * attだけど.....)
        assert self.train_period >= self.infer_period
        if self.training or self.train_period == self.infer_period:

            clipwise_pred = torch.sum(
                torch.sigmoid(self.fix_scale(feat)) * torch.softmax(time_att, dim=-1),
                dim=-1,
            )  # sum((bs, 24, time), -1) -> (bs, 24)
            logits = torch.sum(
                self.fix_scale(feat) * torch.softmax(time_att, dim=-1),
                dim=-1,
            )
        else:
            feat_time = feat.size(-1)
            start = (
                feat_time / 2 - feat_time * (self.infer_period / self.train_period) / 2
            )
            end = start + feat_time * (self.infer_period / self.train_period)
            start = int(start)
            end = int(end)
            feat = feat[:, :, start:end]
            att = torch.softmax(time_att[:, :, start:end], dim=-1)
            # att = att / att.sum(-1)
            clipwise_pred = torch.sum(
                torch.sigmoid(self.fix_scale(feat)) * att,
                dim=-1,
            )
            logits = torch.sum(
                self.fix_scale(feat) * att,
                dim=-1,
            )
            time_att = time_att[:, :, start:end]
        return (
            logits,
            clipwise_pred,
            self.fix_scale(feat),
            time_att,
        )


class AttModel(nn.Module):
    def __init__(
        self,
        backbone="resnet34",
        p=0.5,
        n_mels=224,
        num_class=397,
        train_period=15.0,
        infer_period=5.0,
        in_chans=1,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.logmelspec_extractor = nn.Sequential(
            MelSpectrogram(
                32000,
                n_mels=n_mels,
                f_min=20,
                n_fft=2048,
                hop_length=512,
                normalized=True,
            ),
            AmplitudeToDB(top_db=80.0),
            NormalizeMelSpec(),
        )
        self.backbone = timm.create_model(
            backbone, features_only=True, pretrained=True, in_chans=in_chans
        )
        encoder_channels = self.backbone.feature_info.channels()
        dense_input = encoder_channels[-1]
        self.head = AttHead(
            dense_input,
            p=p,
            num_class=num_class,
            train_period=train_period,
            infer_period=infer_period,
        )

    def forward(self, input):
        feats = self.backbone(input)
        return self.head(feats[-1])


def get_metrics(s_true, s_pred, beta=1):
    s_true = set(s_true)
    s_pred = set(s_pred)
    n, n_true, n_pred = len(s_true.intersection(s_pred)), len(s_true), len(s_pred)

    prec = n / n_pred if n_pred else 0
    rec = n / n_true if n_true else 0
    f1 = (1 + beta ** 2) * prec * rec / (prec * (beta ** 2) + rec) if prec + rec else 0

    return {
        "f1": f1,
        "prec": prec,
        "rec": rec,
        "n_true": n_true,
        "n_pred": n_pred,
        "n": n,
    }


def row_wise_f1_score_micro(y_true, y_pred, threshold=0.5, verbose=False):

    events = y_pred > threshold
    labels = [np.argwhere(event).reshape(-1).tolist() for event in events]
    for i, l in enumerate(labels):
        if len(l) == 0:
            labels[i].append(398)
    t_events = y_true > 1e-6
    t_labels = [np.argwhere(t_event).reshape(-1).tolist() for t_event in t_events]
    for i, l in enumerate(t_labels):
        if len(l) == 0:
            t_labels[i].append(398)
    metrics = pd.DataFrame(
        [
            get_metrics(s_true, s_pred, beta=1)
            for s_true, s_pred in zip(labels, t_labels)
        ]
    )
    metric = metrics.mean()
    # print(metric)
    if verbose:
        return metric
    return metric["f1"]


class ThresholdOptimizer:
    def __init__(self, loss_fn):
        self.coef_ = {}
        self.loss_fn = loss_fn
        self.coef_["x"] = [0.5]

    def _loss(self, coef, X, y):
        ll = self.loss_fn(y, X, coef)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._loss, X=X, y=y)
        initial_coef = [0.5]
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method="nelder-mead"
        )

    def coefficients(self):
        return self.coef_["x"]

    def calc_score(self, X, y, coef, verbose=False):
        return self.loss_fn(y, X, coef, verbose)


class Model(nn.Module):
    def __init__(
        self,
        backbone="resnet34",
        p=0.5,
        n_mels=224,
        num_class=397,
        train_period=15.0,
        infer_period=5.0,
        in_chans=1,
    ):
        super().__init__()
        self.model = AttModel(
            backbone, p, n_mels, num_class, train_period, infer_period, in_chans
        )


def load_model(
    weight_path="",
    backbone_name="resnet34",
    p=0.5,
    n_mels=128,
    num_class=397,
    train_period=30,
    infer_pred=5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(
        backbone_name,
        p=0.5,
        n_mels=128,
        num_class=397,
        train_period=30,
        infer_period=5,
    )
    model = prepare_model_for_inference(model, weight_path).to(device)
    model = model.model
    return model


def prepare_model_for_inference(model, path: Path):
    if not torch.cuda.is_available():
        ckpt = torch.load(path, map_location="cpu")
    else:
        ckpt = torch.load(path)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def prediction_for_clip(test_df: pd.DataFrame, clip: np.ndarray, models):
    dataset = SoundscapeDataset(
        df=test_df,
        clip=clip,
        train_period=30,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probas = []
    targets = []
    for image, row_id, target in tqdm(loader):
        row_id = row_id[0]
        image = image.to(device)
        with torch.no_grad():
            image = models[0].logmelspec_extractor(image)[:, None]
            proba = np.array(
                [model(image)[1].detach().cpu().numpy().reshape(-1) for model in models]
            )
            proba = proba.mean(0)
        probas.append(proba)
        targets.append(target.numpy())
    probas_clip = np.stack(probas)
    targets_clip = np.concatenate(targets)
    return probas_clip, targets_clip


def prediction(test_audios, models_cfg):
    thresholder = ThresholdOptimizer(row_wise_f1_score_micro)
    models = [load_model(**model_cfg) for model_cfg in models_cfg]
    warnings.filterwarnings("ignore")
    probas = []
    targets = []
    for audio_path in test_audios:
        clip, _ = sf.read(audio_path)
        seconds = []
        row_ids = []
        for second in range(5, 605, 5):
            row_id = "_".join(audio_path.name.split("_")[:2]) + f"_{second}"
            seconds.append(second)
            row_ids.append(row_id)

        test_df = pd.DataFrame({"row_id": row_ids, "seconds": seconds})
        probas_clip, targets_clip = prediction_for_clip(
            test_df, clip=clip, models=models
        )
        probas.append(probas_clip)
        targets.append(targets_clip)
    probas = np.concatenate(probas)
    targets = np.concatenate(targets)
    thresholder.fit(probas, targets)
    print(
        f"coef: {thresholder.coefficients()}\n",
        f"{thresholder.calc_score(probas, targets, thresholder.coefficients(), True)}\n",
    )
    return thresholder.coef_, models


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--seed",
        default=2020,
        type=int,
        metavar="SE",
        help="seed number",
        dest="seed",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="1 batch run for debug",
        dest="debug",
    )
    parser.add_argument(
        "--weight_stage1",
        default="resnet34",
    )
    return parser.parse_args()


def main(args):
    pl.seed_everything(args.seed)
    weight_paths = [
        f"../../logs/stage1/{args.weight_stage1}/fold0/lightning_logs/version_0/checkpoints/best_loss.ckpt",
        f"../../logs/stage1/{args.weight_stage1}/fold1/lightning_logs/version_0/checkpoints/best_loss.ckpt",
        f"../../logs/stage1/{args.weight_stage1}/fold2/lightning_logs/version_0/checkpoints/best_loss.ckpt",
        f"../../logs/stage1/{args.weight_stage1}/fold3/lightning_logs/version_0/checkpoints/best_loss.ckpt",
    ]
    models_cfg = [
        dict(
            weight_path=weight_path,
            backbone_name=args.weight_stage1,
            p=0.5,
            n_mels=128,
            num_class=397,
            train_period=30,
            infer_pred=5,
        )
        for weight_path in weight_paths
    ]
    coef, models = prediction(test_audios=valid_all_audios, models_cfg=models_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, (trn_idx, val_idx) in enumerate(
        splitter.split(train_df, y=train_df["primary_label"])
    ):
        model = models[i]
        model.head.infer_period = model.head.train_period
        valid_df_fold = train_df.loc[val_idx, :].reset_index(drop=True)
        for j in trange(len(valid_df_fold)):
            v = valid_df_fold.iloc[j]
            primary_label = v["primary_label"]
            filename = v["filename"]
            filepath = os.path.join(
                "../../input/birdclef-2021/train_short_audio", primary_label, filename
            )
            clip, sample_rate = sf.read(filepath)
            dataset = BirdClef2021Dataset(clip)
            wave = dataset[0]["wave"][None, :].to(device)
            with torch.no_grad():
                image = model.logmelspec_extractor(wave)[:, None]
                preds = model(image)
                pred = preds[1].detach().cpu().numpy().reshape(-1)
                pred_framewise = preds[2].detach().cpu().numpy().squeeze(0)
                pred_att = preds[3].detach().cpu().numpy().squeeze(0)

            npy_filepath = os.path.join(
                f"../../input/birdclef-2021/pseudo_label_stage1_{args.weight_stage1}",
                primary_label,
                filename.split(".")[0],
            )
            os.makedirs(
                os.path.join(
                    f"../../input/birdclef-2021/pseudo_label_stage1_{args.weight_stage1}",
                    primary_label,
                ),
                exist_ok=True,
            )
            np.save(npy_filepath, pred)
            np.save(npy_filepath + "_framewise", pred_framewise)
            np.save(npy_filepath + "_att", pred_att)


if __name__ == "__main__":
    main(get_args())
