# Working in this repo

- Agents work on a branch, never on `main`. The maintainer reviews and merges.
- `main` always builds and passes `go test ./...`.
- Merge with `git merge --ff-only`, rebasing onto `main` first, so history stays linear.
