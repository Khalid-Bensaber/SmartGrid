# SmartGrid Docker Pack v3

Cette version est pensée pour le **travail quotidien** et les **tests proches de la prod**, sans rebuild monstrueux.

## Ce que cette version garantit

- **Une seule image** est buildée : `smartgrid:dev`
- `cli`, `api` et `notebook` **réutilisent la même image**
- Le build reste **léger** car l'image n'embarque pas tout le repo
- Le repo local est monté dans `/workspace` avec `.:/workspace`
- Donc **tout ce qui est dans le repo local reste persistant entre les runs**, notamment :
  - `artifacts/`
  - `data/raw/`
  - `data/interim/`
  - `data/external/`
  - `data/processed/` (déjà versionné dans Git)
  - le code, configs, notebooks, scripts
- Le cache `uv` est conservé dans un volume nommé pour accélérer les redémarrages

## Hypothèse de fonctionnement

Cette solution suppose que tu as **déjà cloné le repo sur la machine hôte** puis que tu déposes ces fichiers à la racine du repo :

```bash
git clone -b dev https://github.com/Khalid-Bensaber/SmartGrid.git
cd SmartGrid
```

Ensuite :

```bash
docker compose build cli --progress=plain
docker compose run --rm cli
docker compose up api
docker compose --profile notebook up notebook
```

## Pourquoi c'est mieux

L'ancienne approche rebuildait et exportait plusieurs services vers le même tag, ce qui était lent et a fini par casser.
Ici :

- `cli` est le seul service qui possède `build:`
- `api` et `notebook` utilisent simplement `image: smartgrid:dev`
- le Dockerfile n'utilise plus `COPY . .`
- les dépendances sont installées séparément avec `uv sync --no-install-project`
- le vrai repo est injecté au runtime par bind mount

## Fichiers

- `Dockerfile` : construit l'image de dev légère
- `docker-compose.yml` : définit `cli`, `api`, `notebook`
- `docker/entrypoint.sh` : prépare les dossiers runtime, resynchronise l'env et lance les checks
- `.dockerignore` : garde le contexte de build petit

## Ce qui reste persistant entre les runs

Parce que le repo complet est monté avec `.:/workspace`, tout ce qui est écrit dans le repo local est conservé :
- nouveaux artefacts de modèle
- exports
- benchmarks
- logs
- données locales
- notebooks modifiés

## Remarque importante

Cette version est la **bonne version pour dev / test / pré-prod pratique**.

Si plus tard tu veux une vraie image autonome de production **sans bind mount du repo**, il faudra faire un deuxième Dockerfile "prod" séparé.
