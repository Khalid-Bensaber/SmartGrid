# SmartGrid Docker Pack v4

Cette version sépare proprement :
- `docker-compose.yml` : base normale, compatible CPU-only
- `docker-compose.gpu.yml` : override optionnel à utiliser seulement sur les machines avec GPU NVIDIA

## Pourquoi cette approche

Tu m'as dit deux choses importantes :
1. certaines machines ont un GPU
2. d'autres n'en auront pas

Donc la meilleure stratégie n'est pas de mettre le GPU "en dur" dans le fichier principal.
Sinon, tu risques des erreurs ou des comportements indésirables sur les machines CPU-only.

## Ce qui reste persistant

Le repo complet est monté avec `.:/workspace`, donc restent persistants entre les runs :
- `artifacts/`
- `data/raw/`
- `data/interim/`
- `data/external/`
- `data/processed/`
- ton code, tes notebooks, tes configs, tes scripts

Le cache `uv` est aussi gardé dans un volume nommé `smartgrid_uv_cache`.

## Usage

### Cas standard (CPU ou machine sans GPU)

```bash
docker compose build cli --progress=plain
docker compose run --rm cli
docker compose up api
```

### Cas machine GPU

Tu gardes le même build, puis tu ajoutes l'override GPU seulement au moment du run/up :

```bash
docker compose build cli --progress=plain
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm cli
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up api
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile notebook up notebook
```

## Fichiers

- `docker-compose.yml` : base standard
- `docker-compose.gpu.yml` : accès GPU NVIDIA optionnel
