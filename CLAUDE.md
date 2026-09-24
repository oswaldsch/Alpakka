# Working in this repo

- Changes spanning several commits, and research tasks, go on a branch that the
  maintainer reviews and merges. A single-commit change may go straight to `main`.
- `main` always builds and passes `go test ./...`.
- Merge with `git merge --ff-only`, rebasing onto `main` first, so history stays linear.
