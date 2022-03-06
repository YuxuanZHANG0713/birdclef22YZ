import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch.utils.data as data

from pathlib import Path

BIRD_CODE = {'afrsil1': 0, 'akekee': 1, 'akepa1': 2, 'akiapo': 3, 'akikik': 4, 
    'amewig': 5, 'aniani': 6, 'apapan': 7, 'arcter': 8, 'barpet': 9, 
    'bcnher': 10, 'belkin1': 11, 'bkbplo': 12, 'bknsti': 13, 'bkwpet': 14, 
    'blkfra': 15, 'blknod': 16, 'bongul': 17, 'brant': 18, 'brnboo': 19, 
    'brnnod': 20, 'brnowl': 21, 'brtcur': 22, 'bubsan': 23, 'buffle': 24, 
    'bulpet': 25, 'burpar': 26, 'buwtea': 27, 'cacgoo1': 28, 'calqua': 29, 
    'cangoo': 30, 'canvas': 31, 'caster1': 32, 'categr': 33, 'chbsan': 34, 
    'chemun': 35, 'chukar': 36, 'cintea': 37, 'comgal1': 38, 'commyn': 39, 
    'compea': 40, 'comsan': 41, 'comwax': 42, 'coopet': 43, 'crehon': 44, 
    'dunlin': 45, 'elepai': 46, 'ercfra': 47, 'eurwig': 48, 'fragul': 49, 
    'gadwal': 50, 'gamqua': 51, 'glwgul': 52, 'gnwtea': 53, 'golphe': 54, 
    'grbher3': 55, 'grefri': 56, 'gresca': 57, 'gryfra': 58, 'gwfgoo': 59, 
    'hawama': 60, 'hawcoo': 61, 'hawcre': 62, 'hawgoo': 63, 'hawhaw': 64, 
    'hawpet1': 65, 'hoomer': 66, 'houfin': 67, 'houspa': 68, 'hudgod': 69, 
    'iiwi': 70, 'incter1': 71, 'jabwar': 72, 'japqua': 73, 'kalphe': 74, 
    'kauama': 75, 'laugul': 76, 'layalb': 77, 'lcspet': 78, 'leasan': 79, 
    'leater1': 80, 'lessca': 81, 'lesyel': 82, 'lobdow': 83, 'lotjae': 84, 
    'madpet': 85, 'magpet1': 86, 'mallar3': 87, 'masboo': 88, 'mauala': 89, 
    'maupar': 90, 'merlin': 91, 'mitpar': 92, 'moudov': 93, 'norcar': 94, 
    'norhar2': 95, 'normoc': 96, 'norpin': 97, 'norsho': 98, 'nutman': 99, 
    'oahama': 100, 'omao': 101, 'osprey': 102, 'pagplo': 103, 'palila': 104, 
    'parjae': 105, 'pecsan': 106, 'peflov': 107, 'perfal': 108, 'pibgre': 109, 
    'pomjae': 110, 'puaioh': 111, 'reccar': 112, 'redava': 113, 'redjun': 114, 
    'redpha1': 115, 'refboo': 116, 'rempar': 117, 'rettro': 118, 'ribgul': 119, 
    'rinduc': 120, 'rinphe': 121, 'rocpig': 122, 'rorpar': 123, 'rudtur': 124, 
    'ruff': 125, 'saffin': 126, 'sander': 127, 'semplo': 128, 'sheowl': 129, 
    'shtsan': 130, 'skylar': 131, 'snogoo': 132, 'sooshe': 133, 'sooter1': 134, 
    'sopsku1': 135, 'sora': 136, 'spodov': 137, 'sposan': 138, 'towsol': 139, 
    'wantat1': 140, 'warwhe1': 141, 'wesmea': 142, 'wessan': 143, 'wetshe': 144, 
    'whfibi': 145, 'whiter': 146, 'whttro': 147, 'wiltur': 148, 'yebcar': 149, 
    'yefcan': 150, 'zebdov': 151}

# BIRD_CODE = {
#     'aldfly': 0, 'ameavo': 1, 'amebit': 2, 'amecro': 3, 'amegfi': 4,
#     'amekes': 5, 'amepip': 6, 'amered': 7, 'amerob': 8, 'amewig': 9,
#     'amewoo': 10, 'amtspa': 11, 'annhum': 12, 'astfly': 13, 'baisan': 14,
#     'baleag': 15, 'balori': 16, 'banswa': 17, 'barswa': 18, 'bawwar': 19,
#     'belkin1': 20, 'belspa2': 21, 'bewwre': 22, 'bkbcuc': 23, 'bkbmag1': 24,
#     'bkbwar': 25, 'bkcchi': 26, 'bkchum': 27, 'bkhgro': 28, 'bkpwar': 29,
#     'bktspa': 30, 'blkpho': 31, 'blugrb1': 32, 'blujay': 33, 'bnhcow': 34,
#     'boboli': 35, 'bongul': 36, 'brdowl': 37, 'brebla': 38, 'brespa': 39,
#     'brncre': 40, 'brnthr': 41, 'brthum': 42, 'brwhaw': 43, 'btbwar': 44,
#     'btnwar': 45, 'btywar': 46, 'buffle': 47, 'buggna': 48, 'buhvir': 49,
#     'bulori': 50, 'bushti': 51, 'buwtea': 52, 'buwwar': 53, 'cacwre': 54,
#     'calgul': 55, 'calqua': 56, 'camwar': 57, 'cangoo': 58, 'canwar': 59,
#     'canwre': 60, 'carwre': 61, 'casfin': 62, 'caster1': 63, 'casvir': 64,
#     'cedwax': 65, 'chispa': 66, 'chiswi': 67, 'chswar': 68, 'chukar': 69,
#     'clanut': 70, 'cliswa': 71, 'comgol': 72, 'comgra': 73, 'comloo': 74,
#     'commer': 75, 'comnig': 76, 'comrav': 77, 'comred': 78, 'comter': 79,
#     'comyel': 80, 'coohaw': 81, 'coshum': 82, 'cowscj1': 83, 'daejun': 84,
#     'doccor': 85, 'dowwoo': 86, 'dusfly': 87, 'eargre': 88, 'easblu': 89,
#     'easkin': 90, 'easmea': 91, 'easpho': 92, 'eastow': 93, 'eawpew': 94,
#     'eucdov': 95, 'eursta': 96, 'evegro': 97, 'fiespa': 98, 'fiscro': 99,
#     'foxspa': 100, 'gadwal': 101, 'gcrfin': 102, 'gnttow': 103, 'gnwtea': 104,
#     'gockin': 105, 'gocspa': 106, 'goleag': 107, 'grbher3': 108, 'grcfly': 109,
#     'greegr': 110, 'greroa': 111, 'greyel': 112, 'grhowl': 113, 'grnher': 114,
#     'grtgra': 115, 'grycat': 116, 'gryfly': 117, 'haiwoo': 118, 'hamfly': 119,
#     'hergul': 120, 'herthr': 121, 'hoomer': 122, 'hoowar': 123, 'horgre': 124,
#     'horlar': 125, 'houfin': 126, 'houspa': 127, 'houwre': 128, 'indbun': 129,
#     'juntit1': 130, 'killde': 131, 'labwoo': 132, 'larspa': 133, 'lazbun': 134,
#     'leabit': 135, 'leafly': 136, 'leasan': 137, 'lecthr': 138, 'lesgol': 139,
#     'lesnig': 140, 'lesyel': 141, 'lewwoo': 142, 'linspa': 143, 'lobcur': 144,
#     'lobdow': 145, 'logshr': 146, 'lotduc': 147, 'louwat': 148, 'macwar': 149,
#     'magwar': 150, 'mallar3': 151, 'marwre': 152, 'merlin': 153, 'moublu': 154,
#     'mouchi': 155, 'moudov': 156, 'norcar': 157, 'norfli': 158, 'norhar2': 159,
#     'normoc': 160, 'norpar': 161, 'norpin': 162, 'norsho': 163, 'norwat': 164,
#     'nrwswa': 165, 'nutwoo': 166, 'olsfly': 167, 'orcwar': 168, 'osprey': 169,
#     'ovenbi1': 170, 'palwar': 171, 'pasfly': 172, 'pecsan': 173, 'perfal': 174,
#     'phaino': 175, 'pibgre': 176, 'pilwoo': 177, 'pingro': 178, 'pinjay': 179,
#     'pinsis': 180, 'pinwar': 181, 'plsvir': 182, 'prawar': 183, 'purfin': 184,
#     'pygnut': 185, 'rebmer': 186, 'rebnut': 187, 'rebsap': 188, 'rebwoo': 189,
#     'redcro': 190, 'redhea': 191, 'reevir1': 192, 'renpha': 193, 'reshaw': 194,
#     'rethaw': 195, 'rewbla': 196, 'ribgul': 197, 'rinduc': 198, 'robgro': 199,
#     'rocpig': 200, 'rocwre': 201, 'rthhum': 202, 'ruckin': 203, 'rudduc': 204,
#     'rufgro': 205, 'rufhum': 206, 'rusbla': 207, 'sagspa1': 208, 'sagthr': 209,
#     'savspa': 210, 'saypho': 211, 'scatan': 212, 'scoori': 213, 'semplo': 214,
#     'semsan': 215, 'sheowl': 216, 'shshaw': 217, 'snobun': 218, 'snogoo': 219,
#     'solsan': 220, 'sonspa': 221, 'sora': 222, 'sposan': 223, 'spotow': 224,
#     'stejay': 225, 'swahaw': 226, 'swaspa': 227, 'swathr': 228, 'treswa': 229,
#     'truswa': 230, 'tuftit': 231, 'tunswa': 232, 'veery': 233, 'vesspa': 234,
#     'vigswa': 235, 'warvir': 236, 'wesblu': 237, 'wesgre': 238, 'weskin': 239,
#     'wesmea': 240, 'wessan': 241, 'westan': 242, 'wewpew': 243, 'whbnut': 244,
#     'whcspa': 245, 'whfibi': 246, 'whtspa': 247, 'whtswi': 248, 'wilfly': 249,
#     'wilsni1': 250, 'wiltur': 251, 'winwre3': 252, 'wlswar': 253, 'wooduc': 254,
#     'wooscj2': 255, 'woothr': 256, 'y00475': 257, 'yebfly': 258, 'yebsap': 259,
#     'yehbla': 260, 'yelwar': 261, 'yerwar': 262, 'yetvir': 263
# }

INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}

PERIOD = 5


class SpectrogramDataset(data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: Path,
                 img_size=224,
                 waveform_transforms=None,
                 spectrogram_transforms=None,
                 melspectrogram_parameters={}):
        self.df = df
        self.datadir = datadir
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["resampled_filename"]
        ebird_code = sample["primary_label"]

        y, sr = sf.read(self.datadir / ebird_code / wav_name)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)
        else:
            len_y = len(y)
            effective_length = sr * PERIOD
            if len_y < effective_length:
                new_y = np.zeros(effective_length, dtype=y.dtype)
                start = np.random.randint(effective_length - len_y)
                new_y[start:start + len_y] = y
                y = new_y.astype(np.float32)
            elif len_y > effective_length:
                start = np.random.randint(len_y - effective_length)
                y = y[start:start + effective_length].astype(np.float32)
            else:
                y = y.astype(np.float32)

        melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)
        melspec = librosa.power_to_db(melspec).astype(np.float32)

        if self.spectrogram_transforms:
            melspec = self.spectrogram_transforms(melspec)
        else:
            pass

        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

        labels = np.zeros(len(BIRD_CODE), dtype=int)
        labels[BIRD_CODE[ebird_code]] = 1

        return {
            "image": image,
            "targets": labels
        }


def mono_to_color(X: np.ndarray,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V
