import regex as re
from multiprocessing import Pool
import heapq
from itertools import chain
from collections import Counter
import pickle
from ..configs.schema import BPETrainerConfig

def BPE_Train(config: BPETrainerConfig):
    num_merges = config.vocab_size - 256 - len(config.special_tokens)
    trainer = BPE_Trainer()
    trainer.ready(
        config.input_path,
        config.special_tokens,
        num_processes=config.num_processes,
    )
    for _ in range(num_merges):
        trainer.merge()
    save_path = config.outputs / config.input_path.stem
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "vocab.pkl", "wb") as f:
        pickle.dump(trainer.vocab,f)
    with open(save_path / "merges.pkl", "wb") as f:
        pickle.dump(trainer.merges,f)


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
TOMBSTONE = None


def pre_tokenize_piece(corpus):
    return re.findall(PAT,corpus)


def _pre_tokenize_batch(segments):
    counts = Counter()
    for segment in segments:
        counts.update(pre_tokenize_piece(segment))
    return counts


def _iter_safe_segments(path, chunksize, special_tokens):
    """
    流式读取文件。
    只在完整的正则匹配边界切分普通文本；
    特殊 token 会被识别并排除，不会被当成普通文本统计。
    """
    if special_tokens:
        special_pattern = "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
        special_pattern = re.compile(special_pattern)
        max_special_len = max(map(len, special_tokens))
    else:
        special_pattern = None
        max_special_len = 1

    buffer = ""
    segment_parts = []
    segment_chars = 0

    def process_matches(matches):
        nonlocal segment_parts, segment_chars

        for match in matches:
            piece = match.group()
            segment_parts.append(piece)
            segment_chars += len(piece)

            # 只在完整的正则匹配结束后切批
            if segment_chars >= chunksize:
                yield "".join(segment_parts)
                segment_parts = []
                segment_chars = 0

    def process_buffer(final=False):
        nonlocal buffer, segment_parts, segment_chars
        consumed = 0
        if special_pattern is not None:
            # 先确定特殊 token 分界，再在普通文本内应用 PAT。
            for match in special_pattern.finditer(buffer):
                if not final and match.start() > len(buffer) - max_special_len:
                    break
                ordinary = buffer[consumed:match.start()]
                if len(ordinary) < chunksize:
                    # 特殊 token 已确定边界；普通片段由工作进程做一次预分词。
                    segment_parts.append(ordinary)
                    segment_chars += len(ordinary)
                else:
                    yield from process_matches(re.finditer(PAT, ordinary))
                if segment_parts:
                    yield "".join(segment_parts)
                    segment_parts = []
                    segment_chars = 0
                consumed = match.end()

        tail = buffer[consumed:]
        matches = list(re.finditer(PAT, tail))
        if final:
            committed = matches
        else:
            cutoff = max(0, len(tail) - max_special_len)
            # 后续字符可能改变末尾的空白或缩写匹配，保留最后两个匹配。
            committed = []
            for match in matches[:-2]:
                if match.end() > cutoff:
                    break
                committed.append(match)
        yield from process_matches(committed)
        if committed:
            consumed += committed[-1].end()
        buffer = buffer[consumed:]
    with open(path, "r", encoding="utf-8") as file:
        while True:
            chunk = file.read(chunksize)
            if not chunk:
                break
            buffer += chunk
            yield from process_buffer()
    # EOF：后面不会再有新文本，处理所有剩余匹配
    yield from process_buffer(final=True)
    if segment_parts:
        final_segment = "".join(segment_parts)
        segment_parts.clear()
        yield final_segment
        del final_segment


def read_and_pretokenize(path,unique_with_count=None,special_tokens=(),num_processes=None,chunksize=4 * 1024 * 1024,):
    """
    分块读取文件，并行 pre-tokenize，汇总每个 pre-token 的频数。
    返回 unique_with_count。
    """
    if unique_with_count is None:
        unique_with_count = {}
    def iter_batches():
        batch = []
        batch_chars = 0
        for segment in _iter_safe_segments(path, chunksize, special_tokens):
            if not segment:
                continue
            batch.append(segment)
            batch_chars += len(segment)
            if batch_chars >= chunksize:
                yield batch
                batch = []
                batch_chars = 0
        if batch:
            yield batch

    batches = iter_batches()
    first_batch = next(batches, None)
    if first_batch is None:
        return unique_with_count
    second_batch = next(batches, None)
    batches = chain((first_batch,), (() if second_batch is None else (second_batch,)), batches)

    def collect(results):
        for counts in results:
            for piece, count in counts.items():
                unique_with_count[piece] = (unique_with_count.get(piece, 0) + count)
    if num_processes == 1 or second_batch is None:
        collect(map(_pre_tokenize_batch, batches))
    else:
        with Pool(processes=num_processes) as pool:
            collect(pool.imap(_pre_tokenize_batch, batches, chunksize=1))
    return unique_with_count


def token_to_bytes(unique_with_count, unique_bytes_with_count):
    for unique_token, count in unique_with_count.items():
        unique_bytes = tuple(bytes([b]) for b in unique_token.encode("utf-8"))
        unique_bytes_with_count[unique_bytes] = count


class BPE_Trainer:
    def __init__(self):
        self.vocab = {b: bytes([b]) for b in range(256)}
        self.merges = []
        self.heap = []
        self.unique_with_count = {}
        self.unique_bytes_with_count = {}
        self.pair_count = {}
        self.pair_to_unique_token = {}
        self.unique_id_to_bytes = []

    def __count_pair__(self):
        for unique_id, (unique_bytes, count) in enumerate(self.unique_bytes_with_count.items()):  
            unique =list(unique_bytes)
            self.unique_id_to_bytes.append([unique,count])
            for idx in range(len(unique_bytes) - 1):
                pair = (unique_bytes[idx], unique_bytes[idx + 1])
                self.pair_count[pair] = self.pair_count.get(pair, 0) + count
                self.pair_to_unique_token.setdefault(pair, set()).add(unique_id)
        del self.unique_bytes_with_count

    def __build_heap__(self):
        self.heap = [(-f, Reverse(pair), pair) for pair, f in self.pair_count.items() if f > 0]
        heapq.heapify(self.heap)

    def __best_pair__(self):
        while self.heap:
            negcount,_,best_pair = self.heap[0]
            if -negcount == self.pair_count[best_pair]:
                return best_pair
            heapq.heappop(self.heap)
        return None

    def merge(self):
        best_pair = self.__best_pair__()
        if best_pair is None:
            return
        may_merge = self.pair_to_unique_token[best_pair]
        temp_update_dict = {}
        for idx in may_merge:
            unique, count = self.unique_id_to_bytes[idx]
            preidx = 0
            idx1 = 0
            idx2 = 0
            while idx2 < len(unique):
                while idx2 < len(unique) and (unique[idx2] is TOMBSTONE or idx2 == idx1):
                    idx2 += 1
                    while idx1 < len(unique) and unique[idx1] is TOMBSTONE:
                        idx1 += 1
                if idx2 >= len(unique) or idx1 >= len(unique):
                    break
                if unique[idx1] == best_pair[0] and unique[idx2] == best_pair[1]:
                    preidx = idx1 - 1
                    while preidx >= 0 and unique[preidx] is TOMBSTONE:
                        preidx -= 1
                    if preidx >= 0:
                        prepair = (unique[preidx], unique[idx1])
                        temp_update_dict[prepair] = (temp_update_dict.get(prepair, 0) - count)
                        newprepair = (unique[preidx],best_pair[0] + best_pair[1])
                        temp_update_dict[newprepair] = (temp_update_dict.get(newprepair, 0) + count)
                        self.pair_to_unique_token.setdefault(newprepair, set()).add(idx)
                    idx3 = idx2
                    unique[idx1] = best_pair[0] + best_pair[1]
                    while idx3 < len(unique) and (idx3 == idx2 or unique[idx3] is TOMBSTONE):
                        idx3 += 1
                    if idx3 != len(unique):
                        sucpair = (unique[idx2],unique[idx3])
                        temp_update_dict[sucpair] = temp_update_dict.get(sucpair, 0) - count
                        newsucpair = (best_pair[0] + best_pair[1],unique[idx3])
                        temp_update_dict[newsucpair] = temp_update_dict.get(newsucpair, 0) + count
                        self.pair_to_unique_token.setdefault(newsucpair, set()).add(idx)

                    unique[idx2] = TOMBSTONE
                    temp_update_dict[best_pair] = temp_update_dict.get(best_pair, 0) - count
                    preidx = idx1
                    idx1 = idx2
                    idx2 = idx3
                else:
                    preidx = idx1
                    idx1 = idx2
                    idx2 += 1
            self.unique_id_to_bytes[idx] = [unique,count]
        for pair, count in temp_update_dict.items():
            new_count = self.pair_count.get(pair,0) + count
            self.pair_count[pair] = new_count
            if new_count > 0:
                heapq.heappush(self.heap, (-new_count, Reverse(pair), pair))

        self.vocab[len(self.vocab)] = best_pair[0] + best_pair[1]
        self.merges.append(best_pair)

    def ready(self,path,special_tokens=(),num_processes=None,chunksize=4 * 1024 * 1024,):
        for token in special_tokens:
            self.vocab[len(self.vocab)] = token.encode("utf-8")
        read_and_pretokenize(path,self.unique_with_count,special_tokens=special_tokens,num_processes=num_processes,chunksize=chunksize,)
        token_to_bytes(self.unique_with_count,self.unique_bytes_with_count,)
        self.__count_pair__()
        self.__build_heap__()
        del self.unique_with_count


class Reverse:
    def __init__(self, pair):
        self.pair = pair
    def __lt__(self, other):
        return self.pair > other.pair


if __name__ == "__main__":
    pass
