# v1.43 stage 1 with retries. Vivado on this machine intermittently fails to read one of its own Tcl files
# ("couldn't read file ...: No error"); a failed attempt leaves the BD half-updated, so each attempt starts from
# the pre-edit backups. Stops at the first attempt that prints BDEDIT_COMPLETE_v1.43.
$top = 'C:\qick_projects\top'
Set-Location $top
$v = 'C:\embedded-software\Vivado\2023.1\bin\vivado.bat'
for ($a = 1; $a -le 4; $a++) {
    Copy-Item -Force 'bd_backups\OG_build_pre_v1.43.bd' 'top.srcs\sources_1\bd\OG_build\OG_build.bd'
    Copy-Item -Force 'top_pre_v1.43.xpr' 'top.xpr'
    $out = "bdedit_v1.43_stdout_a$a.log"
    $p = Start-Process cmd.exe -ArgumentList '/c', "cd /d $top && $v -mode batch -source bdedit_v1.43.tcl -log bdedit_v1.43_a$a.log -journal bdedit_v1.43_a$a.jou > $out 2>&1" -WorkingDirectory $top -WindowStyle Hidden -PassThru
    $p.WaitForExit()
    if (Select-String -Path $out -Pattern 'BDEDIT_COMPLETE_v1.43' -Quiet) { "STAGE1_OK attempt $a"; exit 0 }
    "STAGE1_FAILED attempt ${a}: " + ((Select-String -Path $out -Pattern '^ERROR|^FATAL|couldn.t read file' | Select-Object -First 3 | ForEach-Object { $_.Line }) -join ' | ')
}
"STAGE1_GAVE_UP"
exit 1
