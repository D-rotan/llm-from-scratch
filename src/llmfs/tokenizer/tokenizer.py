import pickle
import regex as re
from typing import Iterable,Iterator
from ..configs.schema import TokenizerConfig
import heapq

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
class Tokenizer:
    def __init__(self, config: TokenizerConfig):
        with open(config.vocab_filepath, "rb") as file_v:
            self.vocab_id_word = pickle.load(file_v)
            self.vocab_word_id = {word:id for id, word in self.vocab_id_word.items()}
            self.token_cache = {}
        if config.merges_filepath is None:
            self.merges = []
        else:
            with open(config.merges_filepath, "rb") as file_m:
                self.merges = pickle.load(file_m)
        self.PAT_REGEX = re.compile(PAT)
        self.special_tokens = config.special_tokens
        special_PAT = "|".join(re.escape(token) for token in sorted(self.special_tokens,key=len,reverse=True))
        self.special_PAT = re.compile(special_PAT)
        self.merges_token = {pair:id for id,pair in enumerate(self.merges)}
    def encode(self,text:str,) -> Iterator[int]:
        segments = self.slice_special_token(text)
        for piece in segments:
            if piece in self.special_tokens:
                yield self.vocab_word_id[piece.encode("utf-8")]
            else:
                for token in self.PAT_REGEX.finditer(piece):
                    token = token.group()
                    ids = self.token_cache.get(token)
                    if ids is not None:
                        yield from ids
                    else:
                        ids_tuple = tuple(self._encode_ordinary(token))
                        self.token_cache[token] = ids_tuple
                        yield from ids_tuple
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    def decode(self, ids: list[int]) -> str :
        text = b"".join(self.vocab_id_word[token_id] for token_id in ids)
        return text.decode("utf-8",errors="replace")
    def slice_special_token(self,text:str) -> Iterator[str]:
        if not self.special_tokens:
            yield text
            return
        consumed = 0
        for match in self.special_PAT.finditer(text):
            if consumed < match.start():
                yield text[consumed:match.start()]
            yield match.group()
            consumed = match.end()
        if consumed < len(text):
            yield text[consumed:]
    # 双向模拟链表 + 堆
    def _encode_ordinary(self,text:str) -> list[int]:
        merges_get = self.merges_token.get
        if not text:
            return []
        def build_heap(text:str):
            char_list = [bytes([b]) for b in text.encode("utf-8")]
            N = len(char_list)
            prev_list = [(i-1) for i in range(N)]
            next_list = [(i+1) for i in range(N)]
            alive_list = [True] * N
            prev_list[0] = -1
            next_list[N - 1] = -1
            heap = []
            for index in range(N - 1):
                rank = merges_get((char_list[index],char_list[index + 1]),-1)
                if rank == -1:
                    continue
                heapq.heappush(heap,(rank,index))
            return char_list,prev_list,next_list,alive_list,heap
        def merge(char_list,prev_list,next_list,alive_list,heap):
            while(heap):
                best_rank,best_idx = heapq.heappop(heap)
                right = next_list[best_idx]
                if right == -1:
                    continue
                if alive_list[best_idx] == False:
                    continue
                bestchar1 = char_list[best_idx]
                bestchar2 = char_list[right]
                if merges_get((bestchar1,bestchar2),-1) != best_rank:
                    continue
                char_list[best_idx] = bestchar1 + bestchar2
                next_list[best_idx] = next_list[right]
                alive_list[right] = False
                if next_list[right] != -1:
                    prev_list[next_list[right]] = best_idx
                prev_list[right] = -1
                next_list[right] = -1
                new_right = next_list[best_idx]
                if new_right != -1:
                    rank = merges_get((char_list[best_idx],char_list[new_right]),-1)
                    if rank != -1:
                        heapq.heappush(heap,(rank,best_idx))
                new_left = prev_list[best_idx]
                if new_left != -1:
                    rank = merges_get((char_list[new_left],char_list[best_idx]),-1)
                    if rank != -1:
                        heapq.heappush(heap,(rank,new_left))
            return char_list,alive_list,prev_list,next_list
        result_list,alive_list,prev_list,next_list = merge(*build_heap(text=text))
        for i in range(len(result_list)):
            head = i
            if alive_list[head] == True and prev_list[head] == -1:
                break
        if head == -1:
            return
        vocab = self.vocab_word_id
        result = []
        while(head != -1):
            result.append(vocab[result_list[head]])
            head = next_list[head]
        return result
