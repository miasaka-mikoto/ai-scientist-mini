# Windows executable

The current build host is Linux, so the checked-in project includes the
reproducible Windows path rather than a renamed non-Windows binary:

```powershell
./build_windows.ps1 -Clean
./dist/AIScientistMini.exe --headless
./dist/AIScientistMini.exe --demo --output demo_output
```

The same steps run in `.github/workflows/build-windows.yml` on a native
`windows-latest` runner and publish `AIScientistMini.exe` as a workflow
artifact. No paid API or external service is required by the build or smoke
test.
