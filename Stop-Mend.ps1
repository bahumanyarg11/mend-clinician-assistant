# Stops what Start-Mend started: the app, the lab SSH tunnel, and the headless Neo4j. (Local Ollama is left running.)
$procs = Get-CimInstance Win32_Process
$targets = @(
    @{ what = "Mend app";   match = { $_.Name -like "python*" -and $_.CommandLine -like "*app_integrated.py*" } },
    @{ what = "lab tunnel"; match = { $_.Name -eq "ssh.exe" -and $_.CommandLine -like "*11435:localhost:11434*" } },
    @{ what = "Neo4j";      match = { $_.Name -eq "java.exe" -and $_.CommandLine -like "*neo4j*" -and $_.CommandLine -like "*console*" } }
)
foreach ($t in $targets) {
    $p = $procs | Where-Object $t.match
    if ($p) { $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; Write-Host "stopped $($t.what)" }
    else { Write-Host "$($t.what) was not running" }
}
