import torch
import torch.nn as nn
from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        all_words = sorted(set(
            word 
            for sent in positive + negative
            for word in sent.split()
            )
        )
        word_dict = {word: idx for idx, word in enumerate(all_words, 1)}
        ans = []
        for sent in positive + negative:
            encoded = []
            for word in sent.split():
                encoded.append(word_dict[word])
            ans.append(torch.Tensor(encoded[:]))
        # print(ans)
        return nn.utils.rnn.pad_sequence(ans, batch_first=True)
        # pass
