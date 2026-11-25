# GROBID PDF parsing (optional)

By default memex turns arXiv PDFs into text with **pdfminer** and splits the text
into ~2,000-character paragraph-aware chunks. That works everywhere, but it
loses the paper's structure: headers, footers and figure captions bleed into the
body, and chunk boundaries ignore sections.

[GROBID](https://github.com/kermitt2/grobid) is a machine-learning parser built
for scholarly PDFs. With it enabled, the arXiv adapter gets:

- **section-aligned chunks** — each chunk stays inside one section and carries
  its section heading, so the Extractor sees "## 3 Results" rather than a
  page-break fragment;
- **the bibliography** — parsed references (title, authors, year, DOI) stored in
  the document's metadata;
- **affiliations** of the authors.

Wikipedia and GDELT documents are unaffected.

## Running it

[`compose.yaml`](../compose.yaml) at the repo root has two variants:

| Service | Image | Size | Runs on | Use when |
|---|---|---|---|---|
| `grobid` | `grobid/grobid:0.9.1-crf` | ~0.5 GB | CPU; amd64 and arm64 (native on Apple Silicon) | default |
| `grobid-gpu` | `grobid/grobid:0.9.1-full` | ~15 GB | NVIDIA GPU; amd64 only | you have a GPU and want the best reference/affiliation parsing |

```bash
docker compose up -d grobid
```

```bash
curl http://localhost:8070/api/isalive   # → true once warm (first start takes a while)
```

Then point memex at it, in `.env` or the shell:

```bash
export MEMEX_GROBID_URL=http://localhost:8070
```

For the GPU variant you need the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
docker compose --profile gpu up -d grobid-gpu
```

The CRF and deep-learning models split sections about equally well; the
deep-learning models mainly improve references and affiliations.

## Behaviour

- `MEMEX_GROBID_URL` unset → pdfminer, no change.
- `MEMEX_GROBID_URL` set → the arXiv adapter checks `/api/isalive` at startup and
  **refuses to start the build if GROBID is unreachable**, rather than silently
  falling back to lower-quality parsing halfway through a build.
- A single PDF GROBID can't parse is skipped; the build continues.
- memex calls `processFulltextDocument` with citation and header consolidation
  turned off, so GROBID never calls out to external services.

On the GPU used for the original builds (an RTX 3090) a warm parse took about
4 seconds per paper; the first call after start-up takes ~30 seconds while
models load. The CPU image has not been timed.
