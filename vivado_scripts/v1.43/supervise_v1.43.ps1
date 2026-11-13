# Supervisor for build_v1.43.tcl. Liveness is file growth, not CPU: the build's stdout and Vivado logs,
# everything under top.runs (runme.log, OOC runs, .Xil realtime/tmp worker dirs) and the project's .Xil.
# 30 min with no write anywhere -> kill this build's process tree and relaunch. Vivado exiting without BUILD_COMPLETE_v1.43 -> relaunch. At most 3 attempts.
# Events go to supervise_v1.43_status.log, one line each.
$ErrorActionPreference = 'Continue'
$top = 'C:\qick_projects\top'
$status = Join-Path $top 'supervise_v1.43_status.log'
$maxAttempts = 3
$silenceLimit = [TimeSpan]::FromMinutes(30)
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -FilePath $status -Append -Encoding utf8 }

function Newest-Write($since) {
    $paths = @("$top\top.runs", "$top\.Xil")
    $newest = $since
    foreach ($p in $paths) {
        if (Test-Path $p) {
            $f = Get-ChildItem -Path $p -Recurse -File -Force -ErrorAction SilentlyContinue |
                 Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($f -and $f.LastWriteTime -gt $newest) { $newest = $f.LastWriteTime }
        }
    }
    foreach ($f in Get-ChildItem -Path $top -File -Filter 'build_v1.43*' -ErrorAction SilentlyContinue) {
        if ($f.LastWriteTime -gt $newest) { $newest = $f.LastWriteTime }
    }
    foreach ($f in Get-ChildItem -Path $top -File -Filter 'v1.43_*' -ErrorAction SilentlyContinue) {
        if ($f.LastWriteTime -gt $newest) { $newest = $f.LastWriteTime }
    }
    return $newest
}

Log "SUPERVISOR start (pid $PID), max $maxAttempts attempts, silence limit 30 min"
for ($a = 1; $a -le $maxAttempts; $a++) {
    $out = "build_v1.43_stdout_a$a.log"
    $cmd = "cd /d $top && C:\embedded-software\Vivado\2023.1\bin\vivado.bat -mode batch -source build_v1.43.tcl -log build_v1.43_a$a.log -journal build_v1.43_a$a.jou > $out 2>&1"
    $p = Start-Process -FilePath cmd.exe -ArgumentList '/c', $cmd -WorkingDirectory $top -WindowStyle Hidden -PassThru
    $t0 = Get-Date
    Log "ATTEMPT $a launched (cmd pid $($p.Id)), stdout $out"
    $lastActive = $t0
    $lastStep = ''
    while ($true) {
        Start-Sleep -Seconds 60
        $nw = Newest-Write $lastActive
        if ($nw -gt $lastActive) { $lastActive = $nw }
        $so = Join-Path $top $out
        if (Test-Path $so) {
            $step = Select-String -Path $so -Pattern '^STEP: ' -ErrorAction SilentlyContinue | Select-Object -Last 1
            if ($step -and $step.Line -ne $lastStep) { $lastStep = $step.Line; Log "A$a $lastStep" }
        }
        $p.Refresh()
        if ($p.HasExited) {
            if ((Test-Path $so) -and (Select-String -Path $so -Pattern 'BUILD_COMPLETE_v1.43' -Quiet)) {
                Log "DONE attempt $a, $([int]((Get-Date) - $t0).TotalMinutes) min"
                exit 0
            }
            $tail = if (Test-Path $so) { (Get-Content $so -Tail 4) -join ' | ' } else { 'no stdout' }
            $fatal = if (Test-Path $so) { (Select-String -Path $so -Pattern '^FATAL|^ERROR' | Select-Object -First 2 | ForEach-Object { $_.Line }) -join ' | ' } else { '' }
            Log "FAILED attempt $a (vivado exited without BUILD_COMPLETE). $fatal || tail: $tail"
            break
        }
        $silent = (Get-Date) - $lastActive
        if ($silent -ge $silenceLimit) {
            if ((Test-Path $so) -and (Select-String -Path $so -Pattern 'BUILD_COMPLETE_v1.43' -Quiet)) {
                Log "HANG in the post-bitstream reports, attempt ${a}; bitstream is complete. Killing build tree $($p.Id), no relaunch"
                & taskkill.exe /PID $p.Id /T /F 2>&1 | ForEach-Object { Log "  $_" }
                exit 0
            }
            Log "HANG attempt ${a}: no log/file growth since $($lastActive.ToString('HH:mm:ss')) ($([int]$silent.TotalMinutes) min); last $lastStep; killing build tree $($p.Id)"
            & taskkill.exe /PID $p.Id /T /F 2>&1 | ForEach-Object { Log "  $_" }
            Start-Sleep -Seconds 10
            break
        }
    }
}
Log "GAVE UP after $maxAttempts attempts"
exit 1
