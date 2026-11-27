> [!IMPORTANT]
>
> Run verification for `irenjj/datafusion` on the local two-node `df-ci` pool and store evidence outside the checkout.
>
> ```bash
> DF_CI=1 scripts/verify-changed-crates.sh "$(git merge-base origin/main HEAD)" ~/df-ci-data/evidence/my-task/
> ```
>
> For on-demand GitHub Actions verification, use `gh workflow run ci.yml -R irenjj/datafusion --ref my-branch`.

## Which issue does this PR close?

<!--
We generally require a GitHub issue to be filed for all bug fixes and enhancements and this helps us generate change logs for our releases. You can link an issue to this PR using the GitHub syntax. For example `Closes #123` indicates that this PR will close issue #123.
Use `Refs #123` for related or partial work and explain what remains.
-->

- Closes #.

## Rationale for this change

<!--
 Why are you proposing this change? If this is already explained clearly in the issue then this section is not needed.
 Explaining clearly why changes are proposed helps reviewers understand your changes and offer better suggestions for fixes.

Please explain the problem you are trying to solve in terms of the user-visible
behavior, rather than the implementation.

For example, "The code in `foo.rs` doesn't handle nulls" is a symptom of the
implementation. "COUNT(DISTINCT) returns wrong results when the column contains
nulls" is the user-visible problem.
-->

## What changes are included in this PR?

<!--
There is no need to duplicate the description in the issue here, but it is sometimes worth providing a summary of the individual changes in this PR.
-->

## What is the testing strategy for this PR?

<!--
We typically require tests for all PRs in order to:
1. Prevent the code from being accidentally broken by subsequent changes
2. Serve as another way to document the expected behavior of the code

Briefly describe how this PR is tested, and point to the specific tests you added. For example: 'This new feature is covered by the `sqllogictest` cases added in `foo.slt`'.

If this PR does not add tests, explain why. For example, if the change is already covered by existing tests, please mention it.

You should also check the `codecov` bot reply on this PR to confirm the changed code is exercised.
-->

### Test evidence

<!--
Record every gate selected by docs/VERIFICATION.md for the change surfaces
touched by this PR. Use a source-bound evidence manifest where available:
scripts/evidence.sh --name <lane> --output-dir ~/df-ci-data/evidence/my-task/ -- <command...>
Record commands that did not run and why instead of leaving rows blank.
-->

| Change surface | Command | Result (exit code) | Evidence JSON path or artifact |
| -------------- | ------- | ------------------ | ------------------------------ |
|                |         |                    |                                |

### Checks not run and why

<!--
List applicable gates that were not run, including TODO targets, unavailable
fixtures, and nightly-only scope. A planned check is not a passing check.
-->

## Are there any user-facing changes?

<!--
If there are user-facing changes then we may require documentation to be updated before approving the PR.

If there are any breaking changes to public APIs, please add the `api change` label.
-->

## Compatibility and rollback impact

<!--
Describe serialized-format, codec, or public API compatibility and how to roll
back. Write "None" only after checking those surfaces.
-->
