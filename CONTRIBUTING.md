# Contributing

Thanks for contributing to `onnxruntime-ep-telum`.

## Ground Rules

- `main` is protected. Do not push directly to `main`.
- All changes must come through pull requests.
- At least one approval is required before merge.
- Code owner review is required for all pull requests.
- CI checks must pass before merge.

## Branching

- Create a feature branch from `main`.
- Keep branch names descriptive and short.
- Rebase/sync with `main` before requesting final review.

## Pull Requests

- Keep PRs focused. Avoid bundling unrelated changes.
- Include a clear summary of what changed and why.
- Add test evidence (logs, CI links, or reproduction steps).
- For EP behavior changes, include model/operator impact and architecture notes.

## Commit Guidance

- Use imperative commit messages (for example, `ci: add gated s390x build job`).
- Keep commits reviewable and logically grouped.

## Ownership and Reviews

- Code owner: `@k8ika0s`.
- Branch protection is configured to require code owner review.
- Branch protection config is versioned in `.github/branch-protection/main.json`.

## Reporting Bugs and Features

- Use the Bug template for runtime/build failures, crashes, and regressions.
- Use the Feature template for new capabilities, provider options, or operator support.
