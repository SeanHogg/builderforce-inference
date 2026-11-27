# Authenticode-signs files with whichever method .github/actions/windows-signing configured in
# the sign config JSON (Azure Trusted Signing `signCommand`, or a .pfx certificate thumbprint).
#   sign-windows.ps1 -Config <sign.json> -Path <dir-or-file> [-Recurse] [-Include *.exe,*.dll]
param(
  [Parameter(Mandatory)] [string] $Config,
  [Parameter(Mandatory)] [string] $Path,
  [switch] $Recurse,
  [string[]] $Include = @('*.exe')
)
$ErrorActionPreference = 'Stop'

$win = (Get-Content $Config -Raw | ConvertFrom-Json).bundle.windows
if (-not $win) { Write-Output '::notice::No Windows signing configured; files stay unsigned.'; exit 0 }

$files = if (Test-Path $Path -PathType Leaf) { Get-Item $Path } else {
  Get-ChildItem $Path -File -Recurse:$Recurse | Where-Object { $n = $_.Name; $Include | Where-Object { $n -like $_ } }
}
if (-not $files) { throw "nothing to sign under $Path" }

$signtool = $null
if (-not $win.signCommand) {
  . (Join-Path $PSScriptRoot 'windows-sdk.ps1')
  $signtool = Get-WindowsSignTool
}

foreach ($f in $files) {
  if ($win.signCommand) {
    Invoke-Expression ($win.signCommand -replace '%1', "`"$($f.FullName)`"")
  } else {
    & $signtool sign /sha1 $win.certificateThumbprint /fd sha256 /tr $win.timestampUrl /td sha256 $f.FullName
  }
  if ($LASTEXITCODE -ne 0) { throw "signing failed for $($f.FullName)" }
}
Write-Output "Signed $(@($files).Count) file(s)."
