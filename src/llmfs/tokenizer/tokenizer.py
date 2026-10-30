import pickle
import regex as re
from typing import Iterable,Iterator
from ..configs.schema import TokenizerConfig
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
class Tokenizer:
    def __init__(self, config: TokenizerConfig):
        with open(config.vocab_filepath, "rb") as file_v:
            self.vocab_id_word = pickle.load(file_v)
            self.vocab_word_id = {word:id for id, word in self.vocab_id_word.items()}
        if config.merges_filepath is None:
            self.merges = []
        else:
            with open(config.merges_filepath, "rb") as file_m:
                self.merges = pickle.load(file_m)
        self.special_tokens = config.special_tokens
        self.merges_token = {pair:id for id,pair in enumerate(self.merges)}
    def _encode_ordinary(self, text: str) -> list[int]:
        text = [bytes([b]) for b in text.encode("utf-8")]
        while(len(text) > 1):
            best_rank = float("inf")
            best_idx = -1
            for idx in range(len(text) - 1):
                pair = (text[idx],text[idx + 1])
                rank = self.merges_token.get(pair,float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = idx
            if best_idx == -1:
                break
            text[best_idx] = text[best_idx] + text[best_idx + 1]
            del text[best_idx + 1]
        text = [self.vocab_word_id[i] for i in text]
        return text
    def encode(self,text:str,) -> list[int]:
        result:list[int] = []
        eachlist = slice_special_token(text,self.special_tokens)
        for piece in eachlist:
            if piece in self.special_tokens:
                result.append(self.vocab_word_id[piece.encode("utf-8")])
            else:
                tokens = re.findall(PAT,piece)
                for token in tokens:
                    result.extend(self._encode_ordinary(token))
        return result
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    def decode(self, ids: list[int]) -> str :
        text = b"".join(self.vocab_id_word[token_id] for token_id in ids)
        return text.decode("utf-8",errors="replace")
def slice_special_token(text:str,special_tokens:list[str]) -> list[str]:
    special_tokens = sorted(special_tokens,key=len,reverse=True)
    if not special_tokens:
        return [text]
    result:list[str] = []
    dislist:list[int] = [0]
    start = 0
    while(True):
        best_idx = float("inf")
        best_token = None
        for special_token in special_tokens:
            special_idx = text.find(special_token,start)
            if best_idx > special_idx and special_idx != -1:
                best_idx = special_idx
                best_token = special_token
        if best_token is None:
            break
        dislist.append(best_idx)
        dislist.append(best_idx + len(best_token))
        start = best_idx + len(best_token)
    dislist = sorted(dislist)
    dislist.append(len(text))
    dislist = list(dict.fromkeys(dislist))
    id = 0
    while(id < len(dislist) - 1):
        result.append(text[dislist[id]:dislist[id + 1]])
        id += 1
    return result


