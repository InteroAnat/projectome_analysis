"""Preserve an exited producer's checkpoint and refresh saved panels; no downloads."""
from datetime import datetime, timezone
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER = ROOT / 'group_analysis/visual_review_20261002/manifest/regional_context_run_20261003.json'
raw = LEDGER.read_bytes()
ledger = json.loads(raw)
if ledger['run_id'] != 'f0a8ad4c5c034ebeb76924873ca1b630':
    raise RuntimeError('Unexpected producer identity')
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.OpenProcess.restype = ctypes.c_void_p
handle = kernel.OpenProcess(0x1000, False, ledger['pid'])
if handle:
    kernel.CloseHandle(ctypes.c_void_p(handle))
    raise RuntimeError('Producer PID exists; refuse recovery writes')
if ctypes.get_last_error() != 87:  # ERROR_INVALID_PARAMETER: PID absent
    raise ctypes.WinError(ctypes.get_last_error())
watcher = json.loads((HERE / 'regional_checkpoint_watcher_20261003.json').read_text())
if watcher['state'] != 'stopped_producer_exited_without_terminal':
    raise RuntimeError('Watcher has not confirmed producer exit')

saved = HERE / ('interrupted_regional_checkpoint_' + ledger['run_id'] + '.json')
if saved.exists() and saved.read_bytes() != raw:
    raise RuntimeError('Preserved checkpoint differs')
saved.write_bytes(raw)
for name in ('successor_regional_readback_20261003.json',
             'regional_checkpoint_watcher_20261003.json',
             'review_artifact_refresh_20261003.json'):
    target = HERE / ('interrupted_' + ledger['run_id'] + '_' + name)
    payload = (HERE / name).read_bytes()
    if target.exists() and target.read_bytes() != payload:
        raise RuntimeError('Preserved audit differs: ' + name)
    target.write_bytes(payload)

audit = json.loads((HERE / 'successor_regional_readback_20261003.json').read_text())
uids = list(dict.fromkeys(audit['already_checked'] + [row['uid'] for row in audit['newly_checked']]))
record = {'observed_utc': datetime.now(timezone.utc).isoformat(),
          'run_id': ledger['run_id'], 'producer_pid_absent': True,
          'cursor_session_exit_code': 1, 'cursor_final_result_received': False,
          'cause': 'unknown; no terminal ledger or final CLI result received',
          'ledger_state_as_saved': ledger['state'], 'ledger_bytes_unchanged': True,
          'preserved_checkpoint': str(saved.relative_to(ROOT)),
          'checkpoint_sha256': hashlib.sha256(raw).hexdigest(),
          'finished_attempts': sum(v['state'] == 'attempt_finished' for v in ledger['identities'].values()),
          'not_attempted': sum(v['state'] == 'not_attempted' for v in ledger['identities'].values()),
          'verified_fields': len(uids), 'network_requests_started': 0,
          'anatomical_acceptance': False, 'repair_state': 'started'}
result_path = HERE / 'interrupted_regional_recovery_20261003.json'
result_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record), flush=True)
commands = [
    [sys.executable, '-B', str(HERE / 'refresh_derived_captions_20261003.py'),
     '--summary-path', str(HERE / 'interrupted_successor_caption_refresh_20261003.json'),
     *[arg for uid in uids for arg in ('--uid', uid)]],
    [sys.executable, '-B', str(HERE / 'verify_completed_regional_20261003.py'),
     '--ledger', str(saved), '--summary-path', str(HERE / 'interrupted_successor_final_readback_20261003.json')],
    [sys.executable, '-B', str(HERE / 'refresh_review_artifacts_20261003.py'),
     '--regional-audit', str(HERE / 'terminal_first_regional_readback_20261003.json'),
     '--regional-audit', str(HERE / 'interrupted_successor_final_readback_20261003.json')],
]
try:
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    if LEDGER.read_bytes() != raw:
        raise RuntimeError('Live ledger changed during recovery')
    record['repair_state'] = 'saved_volume_refresh_and_readback_passed'
except Exception as error:
    record.update(repair_state='failed', error=str(error))
    raise
finally:
    result_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
