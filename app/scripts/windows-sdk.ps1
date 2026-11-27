function Get-WindowsSignTool {
  $tool = Get-Command signtool.exe -CommandType Application -ErrorAction SilentlyContinue
  if ($tool) { return $tool.Source }

  $sdkRoot = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits/10/bin'
  $tool = Get-ChildItem (Join-Path $sdkRoot '*/x64/signtool.exe') -ErrorAction SilentlyContinue |
    Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
  if (-not $tool) { throw "signtool.exe not found under $sdkRoot" }
  return $tool
}
