# ZIP Backup Notes

This file documents how to create a one-file ZIP backup of this repo's current working tree.

The goal is:

- include the real checked-out Git LFS payloads
- include tracked files plus any nonignored untracked files
- exclude gitignored files such as `.venv/`, caches, and other local junk
- exclude `.git/`

## Use This From Repo Root

Run from `/root/hecta-palm`:

```bash
set -euo pipefail
archive_base="hecta-palm-working-tree-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD).zip"
archive_path="/root/${archive_base}"
rm -f "$archive_path"
git ls-files -c -o --exclude-standard -z | xargs -0 zip -q -X "$archive_path"
stat --format='%n %s bytes' "$archive_path"
sha256sum "$archive_path"
```

What this does:

- `git ls-files -c -o --exclude-standard -z`
  - includes tracked files
  - includes untracked but nonignored files
  - excludes gitignored files
- `zip -q -X`
  - writes a ZIP quietly
  - strips extra file attributes for a cleaner, more portable archive

## Important Note About Git LFS

This command zips the current working tree, not Git objects.

That means:

- if the LFS files are already present locally, the ZIP will contain the real large payloads
- if the LFS files have not been pulled, the ZIP will only contain whatever is in the working tree

Before creating the archive, if needed:

```bash
git lfs pull
```

## Why Not `git archive`

Do not use `git archive` for this backup.

For this repo, that is the wrong tool because it does not behave like a working-tree backup:

- it does not include nonignored untracked files
- it can miss the live checked-out LFS payload behavior you want from a full backup blob

## What The ZIP Includes

- current tracked files
- current checked-out LFS payloads
- nonignored untracked files

## What The ZIP Excludes

- `.git/`
- ignored files and directories
- typical examples:
  - `.venv/`
  - `__pycache__/`
  - cache directories
  - any other paths excluded by `.gitignore`

## Verify The Result

After creation:

```bash
ls -lh /root/hecta-palm-working-tree-*.zip
sha256sum /root/hecta-palm-working-tree-*.zip
```

## Transfer To Another Machine

Example `scp` from a local machine:

```bash
scp -P 8884 root@50.35.61.119:/root/hecta-palm-working-tree-<timestamp>-<shortsha>.zip ~/Downloads/
```

## Optional Full Git History Backup

If you need a backup that includes `.git/` and full history, make a separate archive for that.

This ZIP procedure is intentionally a working-tree backup, not a Git-history backup.
