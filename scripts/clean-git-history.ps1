<#
.SYNOPSIS
    Rebuilds the ArchitectOS Git repository with clean history.

.DESCRIPTION
    The repository committed the ArchitectOS Chrome profile: 2,898 of 3,128
    tracked files, including Cookies, Local Storage, IndexedDB and Code
    Cache for every provider signed into. Both existing commits contain it,
    so there is no history worth preserving and a fresh history is simpler
    and safer than surgery on the old one.

    This script:
      1. Verifies you are in the right directory and shows what will happen.
      2. Confirms the new .gitignore excludes everything sensitive.
      3. Deletes .git and re-initializes with a single clean commit.
      4. Stops before pushing so you can inspect the result.

    Your browser profile on disk is NEVER touched. It stays exactly where
    it is with all of your sign-ins intact; it is only removed from Git
    tracking.

.NOTES
    Run from the repository root:
        powershell -ExecutionPolicy Bypass -File scripts\clean-git-history.ps1
#>

$ErrorActionPreference = "Stop"

function Write-Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Write-Ok($text)   { Write-Host "  [ok] $text" -ForegroundColor Green }
function Write-Warn($text) { Write-Host "  [!]  $text" -ForegroundColor Yellow }
function Write-Bad($text)  { Write-Host "  [x]  $text" -ForegroundColor Red }

# ---------------------------------------------------------------------------
# 1. Sanity checks
# ---------------------------------------------------------------------------
Write-Step "Verifying repository"

if (-not (Test-Path ".git")) {
    Write-Bad "No .git directory here. Run this from the repository root."
    exit 1
}
if (-not (Test-Path "src\application.py")) {
    Write-Bad "This does not look like the ArchitectOS repository. Aborting."
    exit 1
}
if (-not (Test-Path ".gitignore")) {
    Write-Bad "No .gitignore. Add it before rebuilding history."
    exit 1
}

$profilePath = "data\chrome-profile"
$profileExists = Test-Path $profilePath
Write-Ok "Repository root confirmed"
if ($profileExists) {
    Write-Ok "Browser profile present at $profilePath (it will NOT be deleted)"
}

# ---------------------------------------------------------------------------
# 2. Confirm the ignore rules actually work
#
# Checked with --no-index because .gitignore does not apply to files that
# are already tracked; without that flag every currently-tracked file
# reports as "not ignored" and the check is meaningless.
# ---------------------------------------------------------------------------
Write-Step "Verifying .gitignore covers the sensitive paths"

$mustIgnore = @(
    "data/chrome-profile/ArchitectOS/Cookies",
    "data/chrome-profile/ArchitectOS/Code Cache/js/sample",
    "data/chrome-profile/ArchitectOS/IndexedDB/x.leveldb/LOCK",
    "data/chrome-profile/ArchitectOS/Local Storage/leveldb/x.ldb",
    "src/__pycache__/config.cpython-313.pyc",
    "data/memory.db",
    "data/diagnostics/shot.png",
    "desktop/node_modules/react/index.js",
    "desktop/src-tauri/target/debug/app.exe",
    ".env"
)

$failed = @()
foreach ($path in $mustIgnore) {
    git check-ignore --no-index -q -- $path 2>$null
    if ($LASTEXITCODE -ne 0) { $failed += $path }
}

if ($failed.Count -gt 0) {
    Write-Bad "These paths are NOT ignored. Fix .gitignore before continuing:"
    $failed | ForEach-Object { Write-Host "        $_" -ForegroundColor Red }
    exit 1
}
Write-Ok "All $($mustIgnore.Count) sensitive patterns are ignored"

# ---------------------------------------------------------------------------
# 3. Preview
# ---------------------------------------------------------------------------
Write-Step "Preview"

$oldCount = (git ls-files | Measure-Object).Count
$gitSize = "{0:N0} MB" -f ((Get-ChildItem .git -Recurse -File -ErrorAction SilentlyContinue |
    Measure-Object -Property Length -Sum).Sum / 1MB)

Write-Host "  Currently tracked files : $oldCount"
Write-Host "  Current .git size       : $gitSize"
Write-Host "  Commits to be discarded : $(git rev-list --count HEAD)"
Write-Host ""
Write-Warn "This deletes .git and creates a new repository with ONE commit."
Write-Warn "All commit history is lost. Your files and browser profile are kept."

$answer = Read-Host "`nType REBUILD to proceed, anything else to abort"
if ($answer -ne "REBUILD") {
    Write-Host "`nAborted. Nothing was changed." -ForegroundColor Yellow
    exit 0
}

# ---------------------------------------------------------------------------
# 4. Rebuild
# ---------------------------------------------------------------------------
Write-Step "Rebuilding repository"

$remote = git remote get-url origin 2>$null
if (-not $remote) {
    $remote = "https://github.com/QUANTUMBABULAL/ArchitectOS.git"
    Write-Warn "No origin found; defaulting to $remote"
}

Remove-Item -Recurse -Force ".git"
Write-Ok "Old history removed"

git init -q
git branch -M main
git remote add origin $remote
Write-Ok "Fresh repository initialized on branch 'main'"

git add .
$newCount = (git diff --cached --name-only | Measure-Object).Count
Write-Ok "Staged $newCount file(s) (was $oldCount)"

# ---------------------------------------------------------------------------
# 5. Final safety gate
#
# Refuse to commit if anything sensitive slipped through. Better to stop
# here than to publish credentials a second time.
# ---------------------------------------------------------------------------
Write-Step "Final safety check"

$staged = git diff --cached --name-only
$dangerous = $staged | Where-Object {
    $_ -match "chrome-profile|Cookies|Local Storage|IndexedDB|Code Cache|node_modules|__pycache__|\.env$|\.db$"
}

if ($dangerous) {
    Write-Bad "Sensitive files are still staged. Refusing to commit:"
    $dangerous | Select-Object -First 20 | ForEach-Object {
        Write-Host "        $_" -ForegroundColor Red
    }
    Write-Host "`n  Fix .gitignore, then run: git reset" -ForegroundColor Yellow
    exit 1
}
Write-Ok "No sensitive files staged"

git commit -q -m "ArchitectOS: multi-provider AI research engine with desktop interface"
Write-Ok "Clean commit created"

# ---------------------------------------------------------------------------
# 6. Hand over
# ---------------------------------------------------------------------------
Write-Step "Done — review, then push"

Write-Host "  Tracked files : $newCount"
Write-Host "  Branch        : main"
Write-Host "  Remote        : $remote"
Write-Host ""
Write-Host "  Inspect first:" -ForegroundColor Cyan
Write-Host "      git ls-files | Select-String 'chrome-profile'   # expect nothing"
Write-Host "      git log --stat"
Write-Host ""
Write-Host "  Then force-push (this REPLACES the remote branch):" -ForegroundColor Cyan
Write-Host "      git push --force -u origin main"
Write-Host ""
Write-Warn "The old 'master' branch still exists on GitHub with the leaked"
Write-Warn "profile. Delete it in the GitHub UI, or run:"
Write-Host "      git push origin --delete master"
Write-Host ""
Write-Warn "Rotate your sessions if you have not already: sign out of all"
Write-Warn "devices on Google, ChatGPT, Grok (x.ai), DeepSeek and Claude."
