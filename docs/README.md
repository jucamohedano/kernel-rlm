# Documentation app development and publication

This is the original RLM framework documentation app, hosted within the independent Kernel RLM repository. Its content is not a public training-metrics dashboard.

From this directory:

```bash
npm ci
npm test
npm run build
npm run dev -- --hostname 127.0.0.1 --port 7780
```

Open `http://127.0.0.1:7780/kernel-rlm/`. The configured project path is `/kernel-rlm`, not the upstream repository's `/rlm`. Static output is written to `out/`.

## CI is separate from publication

Documentation changes on pull requests and `main` are tested and built without requiring an enabled GitHub Pages site. CI attaches the static output as a build-preview artifact. It does not publish it to a public website.

To publish intentionally:

1. Review documentation and repository visibility. Publishing the docs does not sanitize or publish private experiment databases.
2. Enable GitHub Pages in this repository's settings and choose GitHub Actions as the source. Hosting availability depends on repository visibility and account settings; CI does not change either automatically.
3. Manually run **Deploy Docs to GitHub Pages** from the intended reviewed branch with `publish` enabled. The default is build-only.
4. Verify the deployed site at the repository's configured Pages URL, including its `/kernel-rlm/` base path and asset links.

The publication job retains the deployment environment and Pages-specific permissions. Normal PR builds have read-only repository permissions and do not call the Pages configuration/deployment APIs. A missing Pages site should not prevent a documentation preview build.
