param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('chat', 'integration', 'ide')]
    [string]$Mode
)

$baseRoot = "C:\public_html\models\gpt2_medium_dx11\MX2LLM\brain\micronaut\kuhul-es-1.0.18"

switch ($Mode) {
    'chat' {
        $entry = Join-Path $baseRoot 'micronaut-ui-chat-app.ps1'
    }
    'integration' {
        $entry = Join-Path $baseRoot 'mx2_integration.ps1'
    }
    'ide' {
        $entry = Join-Path $baseRoot 'micronaut-ui-chat-app.ps1'
    }
}

if (-not (Test-Path $entry)) {
    throw "Entry point not found: $entry"
}

Write-Host "Launching $Mode builder: $entry"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $entry
