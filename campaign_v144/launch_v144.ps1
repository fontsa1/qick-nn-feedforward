# Starts the v1.44 reacquisition detached; it keeps going if the editor or the terminal session closes.
$bash = "C:\Program Files\Git\bin\bash.exe"
$root = "."
$p = Start-Process -FilePath $bash -ArgumentList "campaign_v144/run_all.sh" -WorkingDirectory $root -WindowStyle Hidden -PassThru
"run_all pid $($p.Id)"
