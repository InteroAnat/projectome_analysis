$PY = 'C:\Users\laika_yan\miniconda3\envs\deb_fmri_pipeline\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$log = 'D:\projectome_analysis\group_analysis\staging_20260926\step1_ingest_log.txt'
function Run-One($sid) {
  Add-Content $log ("==== " + $sid + " start " + (Get-Date -Format o))
  & $PY D:\projectome_analysis\group_analysis\scripts\run_step1_one_sample.py $sid *>> $log
  Add-Content $log ("==== " + $sid + " exit=$LASTEXITCODE " + (Get-Date -Format o))
}
'' | Set-Content $log
foreach ($sid in @('252790','250432','252714')) { Run-One $sid }
Add-Content $log '==== ALL DONE'
