# Reviewed remote code

Some StarVLA backbones and tokenizers ship their own Python code on the Hugging Face Hub
("remote code", loaded with `trust_remote_code=True`). The StarVLA workers never run code from a
model folder or the Hub. They answer every request for such a class from the copies here, and only
for the files listed below. If a model folder carries its own copy of one of these files, it must
match the SHA-256 below, that is, the pinned upstream revision. Any other remote code is refused.

The files are unmodified copies of the upstream files at the pinned revisions, so their hashes match
upstream's; they are excluded from ruff, which would rewrite them.

| File                                  | Upstream                                                                        | Revision                                   | Licence    | SHA-256                                                            |
| ------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------ | ---------- | ------------------------------------------------------------------ |
| `fast/processing_action_tokenizer.py` | [physical-intelligence/fast](https://huggingface.co/physical-intelligence/fast) | `ec4d7aa71691cac0b8bed6942be45684db2110f4` | Apache-2.0 | `6f021ca1f4c1b194ab6fa399d80baf3d642eadb17efb8f73301e4ac401522c20` |

Review notes: the file evaluates no strings, deserializes no pickles, starts no processes, and
opens no network connections or files. It builds QwenFast's FAST action tokenizer (a DCT of the
action chunk, then byte-pair encoding) on top of Transformers' `ProcessorMixin`.

To update a file, review the new upstream revision, copy it unchanged, and update the hash here and in
`REVIEWED_REMOTE_CODE` (`../starvla_worker_common.py`); a unit test checks that they agree.
