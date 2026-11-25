# Notices

## Bundled third-party code

`memex/ui/trace_ui/static/vendor/` ships minified copies of:

| Library | Version | License |
|---|---|---|
| [Alpine.js](https://alpinejs.dev/) | 3.14.9 | MIT |
| [Tailwind CSS](https://tailwindcss.com/) (Play CDN build) | 3.4.17 | MIT |
| [vis-network](https://visjs.github.io/vis-network/) | 9.1.9 | Apache-2.0 or MIT |

Python dependencies are installed from PyPI, not bundled; see
[pyproject.toml](pyproject.toml) and [uv.lock](uv.lock).

The slide deck in [docs/deck/](docs/deck/) loads Space Grotesk, IBM Plex Sans
and IBM Plex Mono from Google Fonts (SIL Open Font License).
Its three full-page illustrations (the 1945 memex desk, the mountain of papers
and the closing scene) were generated with an image model.

## Data sources

memex fetches documents at build time from public services. The text it stores
in your `kg.db` stays under each source's own terms:

- **arXiv** — metadata via the [arXiv API](https://info.arxiv.org/help/api/index.html)
  ([terms of use](https://info.arxiv.org/help/api/tou.html)); paper PDFs remain
  under the license their authors chose, which is often arXiv's non-exclusive
  distribution license and does not allow redistribution.
- **Wikipedia** — article text via the MediaWiki API, under
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- **GDELT** — article search via the
  [GDELT DOC 2.0 API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/);
  the articles themselves belong to their publishers.
- **OpenAlex** — paper metadata under [CC0](https://creativecommons.org/publicdomain/zero/1.0/).

Check those terms before sharing a graph you built. The demo graph
(`memex fetch-demo`) contains only Wikipedia articles and openly licensed arXiv
papers; its dataset card lists every source document and its license.
