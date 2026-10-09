#!/usr/bin/env bash
# Turn the CI workflow into a merge gate: protect master so that
#   - changes reach it only through pull requests (no direct pushes, also for admins),
#   - the "CI gate" check (.github/workflows/ci.yml) must pass on an up-to-date branch,
#   - review conversations must be resolved, and force-pushes / deletion are blocked.
# Also enables "delete branch on merge", which retargets stacked PRs automatically.
#
# Requires the gh CLI, authenticated as a repository admin. Safe to re-run (it replaces
# the rule). Settings can be overridden through the environment:
#   REPO=owner/name  BRANCH=master  APPROVALS=1  tools/protect_branch.sh
set -euo pipefail

REPO="${REPO:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"
BRANCH="${BRANCH:-master}"
APPROVALS="${APPROVALS:-0}"        # raise to 1+ once there are several maintainers
GITHUB_ACTIONS_APP_ID=15368        # only accept "CI gate" when GitHub Actions reports it

gh api --method PUT "repos/$REPO/branches/$BRANCH/protection" --input - <<JSON
{
  "required_status_checks": {
    "strict": true,
    "checks": [{"context": "CI gate", "app_id": $GITHUB_ACTIONS_APP_ID}]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": $APPROVALS,
    "dismiss_stale_reviews": true
  },
  "restrictions": null,
  "required_conversation_resolution": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_linear_history": false
}
JSON

gh repo edit "$REPO" --delete-branch-on-merge >/dev/null

echo "Protected $REPO@$BRANCH: pull requests only, 'CI gate' required (branch up to date),"
echo "approvals required: $APPROVALS, admins included, force-push and deletion blocked."
