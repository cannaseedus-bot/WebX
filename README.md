# Agents.NET V_1

Versioned release folder for the native .NET agent layer.

This package references the live source tree under `../../native/dotnet` and gives SCX control-flow a stable version target for:

- .NET agents and agent runtimes
- provider connectors
- experimental orchestration and process runtimes
- PowerShell-LLM hive model docs
- native .NET architecture docs

## Layout

```text
Agents.NET.V_1/
├── manifest.json
├── README.md
├── registry/
│   └── agents-net.registry.json
└── tools/
    └── validate_release.py
```

## Validation

```powershell
python tools/validate_release.py
```
