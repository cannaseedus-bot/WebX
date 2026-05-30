param(
    [Parameter(Mandatory = $true)]
    [string]$TargetRoot
)

$sourceRoot = "C:\public_html\models\gpt2_medium_dx11\MX2LLM\brain\micronaut\kuhul-es-1.0.18\MX-2"
$lockedManifest = Join-Path $sourceRoot "MICRONAUTS.md"

if (-not (Test-Path $sourceRoot)) {
    throw "MX-2 source folder not found: $sourceRoot"
}

New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null

$snapshot = [ordered]@{
    source = $sourceRoot
    target = $TargetRoot
    locked = $true
    copied = @()
}

Get-ChildItem -Path $sourceRoot -Recurse -File | ForEach-Object {
    $relative = $_.FullName.Substring($sourceRoot.Length).TrimStart('\')
    $destination = Join-Path $TargetRoot $relative
    New-Item -ItemType Directory -Force -Path (Split-Path $destination) | Out-Null
    Copy-Item -LiteralPath $_.FullName -Destination $destination -Force
    $snapshot.copied += $relative
}

$snapshot | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $TargetRoot "micronaut_builder_snapshot.json")

Write-Host "Built Micronaut bundle from locked MX-2 example into $TargetRoot"
Write-Host "Source manifest: $lockedManifest"
