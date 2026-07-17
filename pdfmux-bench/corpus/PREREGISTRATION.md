# GT-0 Corpus — Pre-registration

This file freezes the GT-0 ground-truth corpus **before any pdfmux score was computed on it**, so the false-positive / false-negative numbers in `../VERIFIER-VALIDATION.md` cannot have been obtained by tuning the corpus or the ground truth to flatter the engine.

## Frozen hashes

- **`manifest.json` SHA-256:** `12f3a360d0b02c84cfd8cf62fcb6eabbb12821e3f516777defbea5950b4d6823`
- **Ground-truth content digest** (SHA-256 over all 24 `*.gt.md` files, each as `relpath\0content\0`, sorted by path): `7de1ada8a2173745313cc89152e7808d39134ff105c43664c20a8fc61cbf35a0`

The manifest pins every source PDF by SHA-256 and points at the committed ground-truth file for each document; the digest above pins the ground-truth *content*. Re-deriving either hash from a later checkout proves the corpus is byte-identical to what was frozen here. All sources are static byte-stable files (IRS/arXiv/RFC-Editor/govinfo/USGS/BLS/GAO/LoC-tile/Wikimedia-Commons uploads) — no on-the-fly-rendered endpoints, so the pinned SHA-256 stays reproducible over time.

## Verify

```bash
# from pdfmux-bench/corpus/
shasum -a 256 manifest.json     # must equal the manifest SHA-256 above
python - <<'PY'
import hashlib, pathlib
h = hashlib.sha256()
for f in sorted(pathlib.Path('.').glob('*/*.gt.md')):
    h.update(f.as_posix().encode() + b'\0' + f.read_bytes() + b'\0')
print(h.hexdigest())
PY
```

## Ordering guarantee (hard acceptance criterion)

The commit that adds this file **precedes** the commit that adds `../VERIFIER-VALIDATION.md` (the scoring/validation results). Confirm with:

```bash
git log --oneline --follow pdfmux-bench/corpus/PREREGISTRATION.md
git log --oneline --follow pdfmux-bench/VERIFIER-VALIDATION.md
```

## Provenance rule

Every ground-truth file was authored as **`rendered-image transcription, non-contestant`**: a 200-DPI PyMuPDF render of the benchmarked page(s) was read by a human/non-contestant and transcribed by hand. **No ground truth was derived from any extractor's text output** (not pdfmux's, not any competitor's). Deriving GT from an extractor would make that engine score ≈1.0 by construction; this corpus is authored to be able to catch a wrong verdict.

## Frozen documents (24)

| id | category | pages | source PDF SHA-256 |
|---|---|---|---|
| `irs-f1040` | forms | 1 | `3d31c226df0d189ced80e039d01cf0f8820c1019681a0f0ca6264de277b7e982` |
| `irs-fw9` | forms | 1 | `2d420cbb4123dcf1fb82595b2359cfbb5d81f00b9df9d359fcc7af361d093f53` |
| `irs-fw4` | forms | 1 | `92444d8856ce55d9e25dca8b6d1420634fc68b11e1ab1f760916ea29ddd312b2` |
| `arxiv-1706.03762-attention` | academic | 1-2 | `bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697` |
| `arxiv-1810.04805-bert` | academic | 1 | `5692a5514787a8c6727b4ff3b726a3385798bc68e12138d1d4af83947e2acf6e` |
| `arxiv-1512.03385-resnet` | academic | 1 | `1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a` |
| `arxiv-2203.02155-instructgpt` | complex-tables | 26 | `c1984bb50a5b90fddb895fdc3a0f72e5bc977148c9f63ef6040cbe7a3e1f0d98` |
| `bls-empsit` | complex-tables | 13 | `d7b56ada97d093582421828c89ff02aa6b53ee4d33172a6da5dd8965c7027abb` |
| `gao-24-106214` | complex-tables | 2 | `e001c84f7a59461727c6da1801b2f97c7e0532084b8676809b0e7462b0547757` |
| `rfc1951-deflate` | digital-native | 1-2 | `44ebfc2a0af072f08bc96d68f5f9193ec41332a9e01a46f33b1834d85caced6e` |
| `rfc2616-http` | digital-native | 1 | `4b992042e3a18f1bb8cebbd0916c66b89b4ab179d75d552a85f48f83a222e631` |
| `rfc9110-http-semantics` | digital-native | 1 | `60b30efa1048900833d1758440247fe8ac85a3134f2327388dcb24e07d814c89` |
| `ar-morocco-petitions-44-14` | rtl | 2 | `e283fd1f0c6e0f97b4c8e5a842f2e4b517a6c135ba392a6b32f18a3a2fa62847` |
| `ar-morocco-langculture-04-16` | rtl | 2 | `77d9000d5e59d225c7d4e58b35479c9de9088c5811a0aa63d9a1efdb7ffbcd21` |
| `he-declaration-commentary` | rtl | 3 | `eec175931152cc00c0eb316a7fff1133fdacb5732ce60d802cf58a295c18704d` |
| `telegram-garfield-1881` | scanned | 1 | `bf5f2520e104e50944baa1d624b3afdbcedd857e92b9a2681674ac2c223eed31` |
| `ndl-treaty-report` | scanned | 3 | `c29152739ec8dc84fbff6592c53a1783d3aae43a91c9239b1081515b6671e38e` |
| `ndl-hainan-memo` | scanned | 3 | `01242ad7992a5345a5190b3d97e4925e3b80c5cdcb0bdc94e6d51b4586afb9dd` |
| `letter-peabody-1863` | handwriting | 1 | `f03d530ae5eec6e56db6d2708e21d6bc79366a636ff6e80ad9cba99b8d979c37` |
| `letter-beloved-friend-garrison` | handwriting | 2 | `ee43f4c0aab04098d7d31c004d0b9692b9f666b19c37d93ad896c67351d39bcf` |
| `letter-dear-friend-garrison` | handwriting | 2 | `92834073bb72f118680c45c48935668cb9d0cd07e97361400b9dd3a3fcc2794b` |
| `statute-1-1789` | degraded | 1 | `610adcf3f0a85276801c347e54a5a0cf74dc9fedec40dbdf5d90dd823c2b9df3` |
| `usgs-bul-0002` | degraded | 3 | `a995af6020ac2223dbcc6b0721bb57da229e33604fb75ab37efc32faba09f522` |
| `irish-census-1926` | degraded | 1 | `e9ec69c2b02b27f70b6e7aa3bebe7f4846893db4866b7938e51ec364e389d87f` |

