param([string]$Projeto = 'C:\tc2\Circuit_Scanner_Etapa5_Corrigido\project')
$ErrorActionPreference = 'Stop'
$projetoTreino = [IO.Path]::GetFullPath($Projeto)
$modelosTreino = [IO.Path]::GetFullPath((Join-Path $projetoTreino 'models'))
$novoTreino = Join-Path $PSScriptRoot 'models/circuit_scanner_yolo.pt'
$ativoTreino = Join-Path $modelosTreino 'circuit_scanner_yolo.pt'
$resumoTreino = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'models/active_model_summary.json') -Raw | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $novoTreino -Algorithm SHA256).Hash.ToLower() -ne $resumoTreino.model_sha256) { throw 'O peso do pacote está diferente do esperado.' }
if ((Get-FileHash -LiteralPath $ativoTreino -Algorithm SHA256).Hash.ToLower() -ne $resumoTreino.previous_model_sha256) { throw 'O modelo instalado mudou. Preserve sua versão antes de aplicar manualmente.' }
$backupTreino = [IO.Path]::GetFullPath((Join-Path $modelosTreino ('backup_antes_refinamento_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))))
if (-not $backupTreino.StartsWith($modelosTreino + [IO.Path]::DirectorySeparatorChar)) { throw 'Destino de backup inválido.' }
New-Item -ItemType Directory -Path $backupTreino | Out-Null
foreach ($nomeTreino in @('circuit_scanner_yolo.pt','active_model_summary.json')) {
 $destinoTreino = Join-Path $modelosTreino $nomeTreino
 if (Test-Path -LiteralPath $destinoTreino) { Copy-Item -LiteralPath $destinoTreino -Destination (Join-Path $backupTreino $nomeTreino) }
}
try {
 foreach ($nomeTreino in @('circuit_scanner_yolo.pt','active_model_summary.json')) { Copy-Item -LiteralPath (Join-Path $PSScriptRoot ('models/' + $nomeTreino)) -Destination (Join-Path $modelosTreino $nomeTreino) }
 if ((Get-FileHash -LiteralPath $ativoTreino -Algorithm SHA256).Hash.ToLower() -ne $resumoTreino.model_sha256) { throw 'Falha na conferência dos pesos.' }
} catch {
 foreach ($nomeTreino in @('circuit_scanner_yolo.pt','active_model_summary.json')) { if (Test-Path -LiteralPath (Join-Path $backupTreino $nomeTreino)) { Copy-Item -LiteralPath (Join-Path $backupTreino $nomeTreino) -Destination (Join-Path $modelosTreino $nomeTreino) } }
 throw
}
Write-Host ('Modelo aplicado. Backup: ' + $backupTreino + '. Reabra INICIAR_NA_REDE.bat.')
