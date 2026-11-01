import numpy as np

from sa_asr.sadilar import (clean_ast, clean_sama, decode_alaw, lab_utterances, soap_label, split_of,
                            textgrid_utterances)


def test_clean_sama_keeps_foreign_words_and_drops_tags():
    t = "e a o theogele [foreign_eng + today] ke tla ya [foreign_eng + assignment] ye"
    assert clean_sama(t) == "e a o theogele today ke tla ya assignment ye"
    assert clean_sama("[music] haai [laughter] di tla boa") == "haai di tla boa"
    assert clean_sama("go [foreign_eng + facebook] [foreign_eng + page] ya thobela f_m mahlakung") == \
        "go facebook page ya thobela fm mahlakung"
    assert clean_sama("[?] ga ke kgolwe") is None  # unintelligible part: transcript incomplete
    assert clean_sama("[laughter]") is None


def test_clean_ast():
    assert clean_ast("(u) [int] A. A. dubbel nul een [/int] (/u)") == "A. A. dubbel nul een"
    assert clean_ast("(u) Alma Moake (/u)") == "Alma Moake"
    assert clean_ast("(u) *Le_Roux ($) ekskuus (/$) (/u)") == "Le Roux ekskuus"
    assert clean_ast("(u) Carle+ (/u)") is None  # a cut-off word
    assert clean_ast("[int]") is None


TEXTGRID = '''item []:
item [1]:
class = "IntervalTier"
name = "Orthographic"
intervals: size = 4
intervals [1]:
xmin = 0
xmax = 0.37
text = "0"
intervals [2]:
xmin = 0.37
xmax = 1.94
text = "(u) [int] A. A. dubbel nul een [/int] (/u)"
intervals [3]:
xmin = 1.94
xmax = 3.23
text = "[sta]"
intervals [4]:
xmin = 3.23
xmax = 13.4
text = "[int]"
item [2]:
class = "IntervalTier"
name = "Phonetic"
intervals [1]:
xmin = 0.37
xmax = 1.94
text = "(u) gl a: (/u)"
'''


def test_textgrid_utterances_reads_only_the_orthographic_tier():
    assert textgrid_utterances(TEXTGRID) == [(0.37, 1.94, "(u) [int] A. A. dubbel nul een [/int] (/u)")]
    split = TEXTGRID.replace('text = "(u) [int] A. A. dubbel nul een [/int] (/u)"', 'text = "(u) Alma"').replace(
        'text = "[sta]"', 'text = "Moake (/u)"')
    assert textgrid_utterances(split)[0] == (0.37, 3.23, "(u) Alma Moake (/u)")  # one utterance over two intervals


def test_decode_alaw():
    # A-law silence (0xD5) decodes to near zero; 8 kHz becomes 16 kHz
    out = decode_alaw(bytes([0xD5]) * 8000)
    assert out.dtype == np.float32 and len(out) == 16000 and np.abs(out).max() < 0.001
    loud = decode_alaw(bytes([0xAA, 0x2A]) * 4000, sr_out=8000)  # largest positive and negative codes
    assert loud.max() > 0.9 and loud.min() < -0.9


def test_soap_label_and_split():
    assert soap_label(["eng"]) == "eng"
    assert soap_label(["sot", "eng"]) == "sot"
    assert soap_label(["zul", "eng", "zul"]) == "zul"
    assert soap_label(["tsn", "zul"]) is None
    splits = [split_of(f"speaker{i}") for i in range(2000)]
    assert splits.count("test") > 50 and splits.count("validation") > 50 and splits.count("train") > 1700
    assert split_of("speaker7") == split_of("speaker7")


def test_lab_utterances_plain_htk_and_xwaves():
    assert lab_utterances('(u) sawubona baba (/u)' + chr(10)) == [(0.0, None, '(u) sawubona baba (/u)')]
    htk = chr(10).join(['0 4000000 (u)', '4000000 9000000 ngiyabonga', '9000000 12000000 (/u)'])
    assert lab_utterances(htk) == [(0.0, 1.2, '(u) ngiyabonga (/u)')]
    xw = chr(10).join(['signal x', 'nfields 1', '#', '0.5 121 (u)', '1.5 121 yebo', '2.0 121 (/u)'])
    assert lab_utterances(xw) == [(0.0, 2.0, '(u) yebo (/u)')]
    assert lab_utterances('') == []


def test_nearest_pairs_reused_names_by_folder():
    import sys
    sys.path.insert(0, 'scripts')
    from convert_sadilar import nearest
    grids = ['C/transcriptions/BA001/001.TextGrid', 'C/transcriptions/BA002/001.TextGrid']
    assert nearest('C/speech/alaw/BA002/001.alaw', grids) == grids[1]
    assert nearest('C/speech/alaw/BA003/001.alaw', grids) is None
    assert nearest('C/x/001.alaw', grids[:1]) == grids[0]
    assert nearest('C/x/001.alaw', []) is None
