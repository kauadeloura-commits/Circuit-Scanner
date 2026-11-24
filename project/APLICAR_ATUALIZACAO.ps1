param([string]$Destino = 'C:\tc2\Circuit_Scanner_Etapa5_Corrigido\project')
$ErrorActionPreference = 'Stop'
$origemAtualizacao = [IO.Path]::GetFullPath($PSScriptRoot)
$projetoAtualizacao = (Resolve-Path -LiteralPath $Destino).Path.TrimEnd('\')
if (-not (Test-Path -LiteralPath (Join-Path $projetoAtualizacao 'backend\main.py'))) { throw 'A pasta escolhida não contém o projeto Circuit Scanner.' }
if ($projetoAtualizacao -eq $origemAtualizacao -or $origemAtualizacao.StartsWith($projetoAtualizacao + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Extraia a atualização fora da pasta do projeto antes de aplicar.' }
$backupAtualizacao = Join-Path $projetoAtualizacao ('_backup_v5.47.0_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $backupAtualizacao | Out-Null
function CaminhoProjeto([string]$relativo) {
    $resolvido = [IO.Path]::GetFullPath((Join-Path $projetoAtualizacao $relativo))
    if (-not $resolvido.StartsWith($projetoAtualizacao + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Caminho fora do projeto.' }
    return $resolvido
}
$arquivosAtualizacao = @(foreach ($pastaAtualizacao in @('backend', 'frontend', 'models', 'tools\circuitjs')) {
    $pastaOrigemAtualizacao = Join-Path $origemAtualizacao $pastaAtualizacao
    if (Test-Path -LiteralPath $pastaOrigemAtualizacao) { Get-ChildItem -LiteralPath $pastaOrigemAtualizacao -Recurse -File }
})
foreach ($nomeAtualizacao in @('iniciar_scanner.py','INICIAR_NA_REDE.bat')) { $arquivosAtualizacao += Get-Item -LiteralPath (Join-Path $origemAtualizacao $nomeAtualizacao) }
foreach ($arquivoAtualizacao in $arquivosAtualizacao) {
    $relativoAtualizacao = $arquivoAtualizacao.FullName.Substring($origemAtualizacao.Length + 1)
    $alvoAtualizacao = CaminhoProjeto $relativoAtualizacao
    if (Test-Path -LiteralPath $alvoAtualizacao) {
        $salvoAtualizacao = Join-Path $backupAtualizacao $relativoAtualizacao
        New-Item -ItemType Directory -Path (Split-Path -Parent $salvoAtualizacao) -Force | Out-Null
        Copy-Item -LiteralPath $alvoAtualizacao -Destination $salvoAtualizacao
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $alvoAtualizacao) -Force | Out-Null
    Copy-Item -LiteralPath $arquivoAtualizacao.FullName -Destination $alvoAtualizacao -Force
}
# Apenas arquivos antigos do simulador substituído são arquivados; os dados dos projetos permanecem.
foreach ($obsoletoAtualizacao in @('backend\spice_netlist.py','backend\spice_simulation.py','backend\spice_page.py','backend\production_validation.py','tools\ngspice')) {
    $alvoObsoletoAtualizacao = CaminhoProjeto $obsoletoAtualizacao
    if (Test-Path -LiteralPath $alvoObsoletoAtualizacao) {
        $arquivoBackupAtualizacao = Join-Path $backupAtualizacao ('retirados\' + $obsoletoAtualizacao)
        New-Item -ItemType Directory -Path (Split-Path -Parent $arquivoBackupAtualizacao) -Force | Out-Null
        Move-Item -LiteralPath $alvoObsoletoAtualizacao -Destination $arquivoBackupAtualizacao
    }
}
Write-Host 'Atualização aplicada. Inicie INICIAR_NA_REDE.bat na pasta do projeto.'
Write-Host ('Backup dos arquivos substituídos: ' + $backupAtualizacao)


