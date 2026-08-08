# Scrubbing Leaked Secrets from Git History

**Important:** The original commits contained hardcoded Gmail credentials, Oracle database passwords, and other secrets. Removing them in a new commit is **not enough** — they remain in git history until you rewrite it.

After completing Phase 1, run the steps below **before pushing to any public remote** (or force-push if the repo is already public).

## Prerequisites

1. **Back up the repository** (clone to a safe location).
2. **Rotate all leaked credentials immediately:**
   - Revoke the Gmail app password and create a new one (or disable app passwords).
   - Change Oracle DB passwords if those databases were ever reachable.
3. Install [git-filter-repo](https://github.com/newren/git-filter-repo):

   ```powershell
   pip install git-filter-repo
   ```

## Option A: Replace known secret strings (recommended)

From the repository root, create a `secrets.txt` file listing every leaked string (one per line). **Do not commit this file.**

```text
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
```

Then run:

```powershell
cd E:\Life-detection-in-Disaster-zones

git filter-repo --replace-text secrets.txt --force
```

`--force` is required when filter-repo detects an existing clone (not a fresh mirror).

## Option B: Remove specific files from all history

If you prefer to strip the old credential-bearing files entirely:

```powershell
cd E:\Life-detection-in-Disaster-zones

# Historical paths (before rename):
git filter-repo --path "TOC Project/src/alert_message.py" --invert-paths --force
git filter-repo --path "TOC Project/src/database_integration.py" --invert-paths --force
git filter-repo --path "TOC Project/src/image_database_integration.py" --invert-paths --force

# Or current paths (after Phase 1 rename):
# git filter-repo --path "disaster-response-vision/src/alert_message.py" --invert-paths --force
```

Then restore the sanitized versions from your current branch.

## Option C: BFG Repo-Cleaner (alternative)

```powershell
# Download bfg.jar from https://rtyley.github.io/bfg-repo-cleaner/
java -jar bfg.jar --replace-text secrets.txt
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

## After scrubbing

1. Verify secrets are gone:

   ```powershell
   git log --all -p | Select-String -Pattern "yzdk|***REMOVED***|***REMOVED***|akshayofficialnew"
   ```

   This should return **no matches**.

2. Force-push all branches and tags (coordinate with collaborators first):

   ```powershell
   git push origin --force --all
   git push origin --force --tags
   ```

3. Ask anyone who cloned the old repo to re-clone or reset their local copy.

4. Delete `secrets.txt` from disk.

## Note on large binary history

This repository also committed model weights (`*.pt`) and sample media. Phase 1 removes them from tracking going forward. To purge them from history as well (recommended before open-sourcing):

```powershell
git filter-repo --strip-blobs-bigger-than 1M --force
```

Or remove specific paths:

```powershell
git filter-repo --path-glob '*.pt' --invert-paths --force
git filter-repo --path-glob '*.mp4' --invert-paths --force
```
