@echo off
setlocal

set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%"

cmake -S . -B build
cmake --build build --config Release

popd
endlocal
