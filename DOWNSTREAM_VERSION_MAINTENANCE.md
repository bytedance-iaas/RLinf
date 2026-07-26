# Downstream Version Maintenance

## Remotes and branches

- `upstream`: `RLinf/RLinf`
- `origin`: `bytedance-iaas/RLinf`
- `origin/main`: exact mirror of `upstream/main`. Do not add commits or merge
  pull requests directly into this branch.
- `origin/dev`: integration branch for all downstream `feat/*` pull requests.
  Its history is the current downstream patch series rebased onto `origin/main`.
- `origin/release`: CI release branch based on `upstream/release/v0.3`. Only
  reviewed downstream patches are cherry-picked from `dev`.

Release tags are immutable. Rewriting `dev` or `release` must not move an
existing release tag.

## Sync `main`

Fetch both repositories and update the downstream mirror:

```bash
git fetch --all --prune
git push --force-with-lease origin upstream/main:main
```

Verify that both remote-tracking refs resolve to the same commit:

```bash
git fetch origin main
test "$(git rev-parse origin/main)" = "$(git rev-parse upstream/main)"
```

## Sync `dev`

Create a backup ref, rebase the downstream patch series onto the mirrored
`main`, validate it, and update the remote with a lease:

```bash
git fetch --all --prune
git branch backup/dev-YYYYMMDD origin/dev
git switch dev
git rebase origin/main
# Run the relevant lint, unit, documentation, and downstream CI checks.
git push --force-with-lease origin dev
```

All downstream feature pull requests target `dev`; do not merge `dev` into
`main`.

## Maintain `release`

Configure the local release branch to pull/rebase from the current upstream
release line while pushing to the downstream repository:

```bash
git branch --set-upstream-to=upstream/release/v0.3 release
git config branch.release.pushRemote origin
```

Bring reviewed patches over from `dev` individually so their source remains
traceable:

```bash
git switch release
git cherry-pick -x <dev-patch-commit>
```

To update the release base:

```bash
git fetch --all --prune
git branch backup/release-YYYYMMDD origin/release
git switch release
git rebase upstream/release/v0.3
# Run release CI and smoke tests.
git push --force-with-lease origin release
```

Prefer rebase so the branch remains a linear upstream release plus downstream
patch series. Use a merge only when release commits are externally pinned and
history rewriting is prohibited, or when a risky conflict resolution requires
an explicit merge record. Evaluate every existing private patch against the new
upstream release before retaining it.
