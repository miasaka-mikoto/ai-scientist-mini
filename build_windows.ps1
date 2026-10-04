param(
    [string]$Python = "python",
    [switch]$Clean
)
$ErrorActionPreference = "Stop"
if ($Clean) {
    if (Test-Path dist) { Remove-Item -Recurse -Force dist }
    if (Test-Path build) { Remove-Item -Recurse -Force build }
}
& $Python -m PyInstaller --noconfirm --clean --onefile --windowed --name AIScientistMini `
  --paths src `
  --hidden-import aiscientist.core `
  --hidden-import aiscientist.core.models `
  --hidden-import aiscientist.core.analysis `
  --hidden-import aiscientist.core.providers `
  --hidden-import aiscientist.core.store `
  --hidden-import aiscientist.core.engine `
  --hidden-import aiscientist.core.demo `
  --hidden-import aiscientist.reporting `
  --hidden-import aiscientist.visualization `
  --exclude-module scipy --exclude-module matplotlib --exclude-module reportlab `
  --exclude-module PySide6 --exclude-module PIL --exclude-module pandas `
  launcher.py
Write-Host "Built executable: dist/AIScientistMini.exe"
