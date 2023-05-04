from typing import Dict, List, Tuple
import codecs
from text_unidecode import unidecode
import pandas as pd
import re


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end


def resolve_encodings_and_normalize(text: str):
    # Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
    codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
    codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text

def cleanup_text(text:str) -> str:
    punctuation_signs = list("?:!,;")
    text = text.replace("\n","")
    text = text.replace("\r", " ")
    text = text.replace("    ", " ")
    text = text.replace('"', '')
    text = text.replace("'s", "")
    for punct_sign in punctuation_signs:
      text = text.replace(punct_sign, '')
    words = re.sub(pattern = '[^a-zA-Z]',repl = ' ', string = text)
    words = words.lower()
    return words

def text_to_char_idx(full_text:str, substring:str):
    try:
        words = substring.split()
        essay_token = full_text.split()
        substring_text = " ".join(words)
        essay_text = " ".join(essay_token)
        essay_len = len(essay_text)
        start_char = essay_text.find(substring_text)
        end_char = start_char + len(substring_text)
    except Exception as e:
        print(e, full_text, substring)
    return start_char, end_char, essay_len


def data_preprocessing(df:pd.DataFrame,essay_text:str) -> pd.DataFrame:
    df['normalized_discourse'] = df['discourse_text'].apply(resolve_encodings_and_normalize)
    df['processed_discourse'] = df['normalized_discourse'].apply(cleanup_text)
    normalized_essay = resolve_encodings_and_normalize(essay_text)
    processed_essay = cleanup_text(normalized_essay)
    df[['discourse_start','discourse_end','essay_len']]= df.apply(lambda x: text_to_char_idx(processed_essay ,x['processed_discourse']),\
                                                                axis=1 , result_type='expand')
    df.drop(columns =['normalized_discourse'], inplace = True)
    return df