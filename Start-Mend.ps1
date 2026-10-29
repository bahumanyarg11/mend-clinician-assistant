# Mend clinician assistant -- one-click start: Neo4j (headless, no Neo4j Desktop window), the LLM, then the app.
# Double-click Start-Mend.bat (it runs this). Stop everything with Stop-Mend.bat.
# LLM choice (set MEND_LLM in .env):  auto (default) = lab gemma4:26b over an SSH tunnel if the lab answers, else a local model
#                                     lab = lab only      local = laptop Ollama only.   LLM_MODEL in .env overrides the model name.
$Here      = $PSScriptRoot
$LocalPreference = @("gemma4:12b", "qwen3.5:9b", "llama3.1:8b")

function PortOpen($port) { try { $c = New-Object Net.Sockets.TcpClient; $c.Connect("127.0.0.1", $port); $c.Close(); $true } catch { $false } }
function WaitPort($port, $secs) { for ($i = 0; $i -lt $secs; $i += 2) { if (PortOpen $port) { return $true }; Start-Sleep 2 }; $false }
function Fail($msg) { Write-Host "  $msg" -ForegroundColor Red; Read-Host "Press Enter to close"; exit 1 }

# .env -> this process's environment
if (Test-Path "$Here\.env") {
    Get-Content "$Here\.env" | Where-Object { $_ -match '^\s*[A-Za-z_][A-Za-z0-9_]*\s*=' } | ForEach-Object {
        $k, $v = $_ -split '=', 2; [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), "Process") }
}
# NEO4J_HOME / MEND_LAB in .env override these (the defaults are the original dev laptop and lab)
$Neo4jHome = if ($env:NEO4J_HOME) { $env:NEO4J_HOME } else { "$env:USERPROFILE\.Neo4jDesktop2\Data\dbmss\dbms-f4485f74-5531-4bb0-84a4-7c14f8101fcf" }
$Lab       = if ($env:MEND_LAB) { $env:MEND_LAB } else { "atr_ir@10.177.29.11" }
if (-not $env:NEO4J_PASSWORD -or $env:NEO4J_PASSWORD -eq "your_password_here") { Fail "Put your Neo4j password in $Here\.env (NEO4J_PASSWORD=...) first." }

Write-Host "[1/3] Neo4j knowledge graph"
if (PortOpen 7687) { Write-Host "  already running" }
else {
    if (-not (Test-Path "$Neo4jHome\bin\neo4j.bat")) { Fail "Neo4j not found at $Neo4jHome" }
    Start-Process -FilePath "$Neo4jHome\bin\neo4j.bat" -ArgumentList "console" -WorkingDirectory $Neo4jHome -WindowStyle Minimized
    if (WaitPort 7687 120) { Write-Host "  started (runs in a minimised window; Stop-Mend.bat stops it)" } else { Fail "did not start -- see $Neo4jHome\logs\neo4j.log" }
}

Write-Host "[2/3] LLM"
$mode = if ($env:MEND_LLM) { $env:MEND_LLM.ToLower() } else { "auto" }
$useLab = $false
if ($mode -ne "local") {
    if (PortOpen 11435) { $useLab = $true; Write-Host "  lab tunnel already open" }
    else {
        Write-Host "  checking the lab ($Lab)..."
        $tags = ssh -o BatchMode=yes -o ConnectTimeout=10 $Lab "curl -s -m 8 http://localhost:11434/api/tags" 2>$null
        if ("$tags" -match "gemma4:26b") {
            Start-Process ssh -ArgumentList "-N -o BatchMode=yes -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -L 11435:localhost:11434 $Lab" -WindowStyle Minimized
            $useLab = WaitPort 11435 30
        }
    }
}
if ($useLab) {
    $env:OLLAMA_HOST = "http://localhost:11435"
    if (-not $env:LLM_MODEL) { $env:LLM_MODEL = "gemma4:26b" }
    Write-Host "  using the lab: $env:LLM_MODEL"
}
else {
    if ($mode -eq "lab") { Fail "lab not reachable (MEND_LLM=lab). Check ZeroTier, or set MEND_LLM=auto in .env." }
    if (-not (PortOpen 11434)) {
        Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList "serve" -WindowStyle Hidden
        if (-not (WaitPort 11434 30)) { Fail "local Ollama did not start" }
    }
    $env:OLLAMA_HOST = "http://localhost:11434"
    if (-not $env:LLM_MODEL) {
        $have = (Invoke-RestMethod http://localhost:11434/api/tags).models.name
        $env:LLM_MODEL = $LocalPreference | Where-Object { $have -contains $_ } | Select-Object -First 1
        if (-not $env:LLM_MODEL) { Fail "no local model found; run: ollama pull llama3.1:8b" }
    }
    Write-Host "  lab not used -- local model: $env:LLM_MODEL"
}
$env:LLM_BACKEND = "ollama"

Write-Host "[3/3] Mend at http://localhost:8502  (Ctrl+C here or Stop-Mend.bat to stop)"
Set-Location $Here
python -m streamlit run part1_conversational\app_integrated.py --server.port 8502
