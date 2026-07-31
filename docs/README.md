# Website

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Installation

```bash
npm install
```

**Note**: feel free to use the package manager of your choice.

**Python dependency**: the API reference under `docs/api/` is auto-generated at build/start time
(`gen-api` script) from the `src/` docstrings via a headless Sphinx build
(`sphinx` + `sphinx-markdown-builder`, source in `api_src/`) — install these into the same
environment used to run `npm run build`/`start`:

```bash
pip install -e "..[docs]"
```

The example pages under `docs/docs/documentation/examples/` are **not** auto-generated
(Docusaurus can't run Python) — their code/output/plot are pasted in by hand from the generator
scripts in the repo-root `examples/` directory (a sibling of `docs/`, alongside `notebooks/`). If
`examples/lin_elastic_strain.py` changes, re-run it (`npm run gen-example`, or `python
examples/pff_damage.py` for the phase-field page) and update the corresponding page and
`static/img/*.png` manually.

## Local Development

```bash
npm run start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

Using SSH:

```bash
USE_SSH=true npm run deploy
```

Not using SSH:

```bash
GIT_USER=<Your GitHub username> npm run deploy
```

If you are using GitHub Pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
