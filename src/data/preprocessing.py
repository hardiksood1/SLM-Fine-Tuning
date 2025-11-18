from torch.utils.data import Dataset
import torch

class ConcatenatedDataset(Dataset):
    def __init__(self, tokenized_chunks, seq_length):
        """
        tokenized_chunks: List of lists, where each list is token IDs.
        seq_length: Fixed sequence length for training.
        """
        self.data = []
        buffer = []
        current_len = 0

        for chunk in tokenized_chunks:
            # chunk is already a list of input_ids
            buffer.extend(chunk)
            current_len += len(chunk)

            # Cut into blocks of seq_length
            while current_len >= seq_length:
                self.data.append(buffer[:seq_length])
                buffer = buffer[seq_length:]
                current_len -= seq_length

        # Handle the last remaining buffer (pad it)
        if buffer:
            padding_needed = seq_length - len(buffer)
            self.data.append(buffer + [0] * padding_needed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = torch.tensor(self.data[idx], dtype=torch.long)
        return {
            "input_ids": ids,
            "labels": ids.clone()
        }
