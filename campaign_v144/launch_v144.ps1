# Starts campaign_v144/run_all.sh detached, so it keeps running if the terminal closes.
$bash = "C:\Program Files\Git\bin\bash.exe"
$root = "."
$p = Start-Process -FilePath $bash -ArgumentList "campaign_v144/run_all.sh" -WorkingDirectory $root -WindowStyle Hidden -PassThru
"run_all pid $($p.Id)"
