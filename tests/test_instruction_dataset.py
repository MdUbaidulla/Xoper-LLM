import json

from data.instruction_dataset import DPODataset, SFTDataset, dpo_collate, sft_collate
from data.tokenizer_train import train_tokenizer


def _tiny_tokenizer(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world how are you today fine thank you " * 100, encoding="utf-8")
    out_path = str(tmp_path / "tok.json")
    train_tokenizer(str(corpus), vocab_size=300, out_path=out_path)
    from tokenizers import Tokenizer
    return Tokenizer.from_file(out_path)


def test_sft_dataset_masks_prompt_tokens(tmp_path):
    tok = _tiny_tokenizer(tmp_path)
    jsonl = tmp_path / "sft.jsonl"
    rows = [{"prompt": "hello world", "response": "fine thank you"}]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    ds = SFTDataset(str(jsonl), tok, max_len=64)
    assert len(ds) == 1
    ex = ds[0]
    # some prefix of the sequence (the prompt + bos) must be unmasked (loss_mask == 0)
    assert ex.loss_mask[0] == 0
    # some suffix (the response) must be masked-in for loss
    assert ex.loss_mask[-1] == 1
    assert sum(ex.loss_mask) > 0
    assert sum(ex.loss_mask) < len(ex.loss_mask)


def test_sft_collate_shapes_and_padding(tmp_path):
    tok = _tiny_tokenizer(tmp_path)
    jsonl = tmp_path / "sft.jsonl"
    rows = [
        {"prompt": "hello", "response": "fine thank you"},
        {"prompt": "hello world how are you today", "response": "fine"},
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ds = SFTDataset(str(jsonl), tok, max_len=64)

    input_ids, targets, loss_mask = sft_collate([ds[0], ds[1]], pad_id=0)
    assert input_ids.shape == targets.shape == loss_mask.shape
    assert input_ids.shape[0] == 2
    assert loss_mask.max() == 1.0
    assert loss_mask.min() == 0.0


def test_dpo_dataset_and_collate(tmp_path):
    tok = _tiny_tokenizer(tmp_path)
    jsonl = tmp_path / "dpo.jsonl"
    rows = [{"prompt": "hello world", "chosen": "fine thank you", "rejected": "how are you"}]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    ds = DPODataset(str(jsonl), tok, max_len=64)
    assert len(ds) == 1

    chosen, rejected = dpo_collate([ds[0]], pad_id=0, max_len=64)
    c_ids, c_tgt, c_mask = chosen
    r_ids, r_tgt, r_mask = rejected
    assert c_ids.shape == c_tgt.shape == c_mask.shape
    assert r_ids.shape == r_tgt.shape == r_mask.shape
    assert c_mask.sum() > 0
    assert r_mask.sum() > 0
