"""Parse audited R sources and extracted Rmd chunks; never run analyses."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RSCRIPT = Path('C:/Program Files/R/R-4.6.1/bin/Rscript.exe')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_chunks(text):
    lines = text.splitlines()
    extracted, active, opening, count, adjacent = [], False, '', 0, []
    for number, line in enumerate(lines, 1):
        match = re.match(r'^\s*(`{3,}|~{3,})\s*\{\s*[rR](?=[\s,}])',line)
        if match:
            if active:
                adjacent.append(number)
            active, opening = True, match.group(1)
            count += 1
            extracted.append('')
        elif not active:
            extracted.append('')
        elif re.match(r'^\s*'+re.escape(opening)+r'\s*$',line):
            active = False
            extracted.append('')
        else:
            extracted.append(line)
    if active:
        raise ValueError('Unclosed R Markdown chunk')
    return '\n'.join(extracted)+'\n', count, adjacent


R_DRIVER = r'''
args <- commandArgs(trailingOnly=TRUE)
manifest <- read.delim(args[1],stringsAsFactors=FALSE,check.names=FALSE,fileEncoding="UTF-8")
options(knitr.duplicate.label="allow")
has_knitr <- requireNamespace("knitr",quietly=TRUE)
results <- vector("list",nrow(manifest))
try_parse <- function(path) {
  warnings <- character()
  value <- tryCatch(withCallingHandlers({
    parse(file=path,encoding="UTF-8",keep.source=TRUE)
    "pass"
  },warning=function(w) {warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")}),
  error=function(e) conditionMessage(e))
  list(status=if (identical(value,"pass")) "pass" else "fail",error=if (identical(value,"pass")) "" else value,warnings=paste(warnings,collapse=" | "))
}
for (i in seq_len(nrow(manifest))) {
  row <- manifest[i,]
  primary <- try_parse(row$parse_path)
  purl_status <- "not_applicable"
  purl_error <- ""
  if (row$kind=="Rmd" && has_knitr) {
    tangle <- tempfile(fileext=".R")
    purl_status <- tryCatch({
      knitr::opts_chunk$set(eval=FALSE,cache=FALSE)
      knitr::purl(row$source_path,output=tangle,documentation=0,quiet=TRUE)
      check <- try_parse(tangle)
      purl_error <- check$error
      check$status
    },error=function(e) {purl_error <<- conditionMessage(e);"fail"})
    unlink(tangle)
  } else if (row$kind=="Rmd") purl_status <- "knitr_unavailable"
  results[[i]] <- data.frame(id=row$id,status=primary$status,error=primary$error,warnings=primary$warnings,
                            purl_status=purl_status,purl_error=purl_error,stringsAsFactors=FALSE)
}
write.table(do.call(rbind,results),file=args[2],sep="\t",quote=TRUE,row.names=FALSE,na="")
writeLines(c(R.version.string,paste0("knitr_available=",has_knitr)),args[3])
'''


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--override-root',type=Path)
    parser.add_argument('--receipt',default='R_syntax_validation_20261009.json')
    args=parser.parse_args()
    if args.override_root:
        args.override_root=args.override_root.resolve()
    assert Path(args.receipt).name==args.receipt and args.receipt.endswith('.json')
    inventory = json.loads((HERE/'independent_consumer_measurements_20261009.json').read_text())['coverage']
    canonical = [ROOT/item['path'] for item in inventory]
    files = [args.override_root/path.relative_to(ROOT)
             if args.override_root and (args.override_root/path.relative_to(ROOT)).is_file() else path
             for path in canonical]
    assert len(files)==63 and all(path.is_file() for path in files)
    assert RSCRIPT.is_file()
    records = []
    with tempfile.TemporaryDirectory(prefix='r-syntax-only-',dir=HERE) as tmp:
        temp = Path(tmp)
        driver = temp/'parse_only.R'; driver.write_text(R_DRIVER,encoding='utf-8')
        rows = []
        for i,path in enumerate(files):
            record = {'path':str(canonical[i].relative_to(ROOT)),
                      'validated_source_path':str(path.relative_to(ROOT)),
                      'sha256':sha(path),'kind':path.suffix[1:]}
            # R's inherited Windows locale can corrupt Unicode pathnames.
            # Byte-identical ASCII paths isolate syntax from that I/O issue.
            source_copy = temp/f'source_{i:03d}{path.suffix}'
            source_copy.write_bytes(path.read_bytes())
            assert sha(source_copy)==record['sha256']
            parse_path = source_copy
            if path.suffix.lower()=='.rmd':
                code,count,adjacent = extract_chunks(path.read_text(encoding='utf-8-sig'))
                parse_path = temp/f'chunks_{i:03d}.R'
                parse_path.write_text(code,encoding='utf-8')
                record.update(R_chunks=count,all_chunks_extraction_sha256=sha(parse_path),
                              consecutive_unclosed_R_header_lines=adjacent)
            records.append(record)
            rows.append({'id':i,'source_path':str(source_copy),'parse_path':str(parse_path),'kind':record['kind']})
        manifest = temp/'sources.tsv'
        with manifest.open('w',encoding='utf-8',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=['id','source_path','parse_path','kind'],delimiter='\t')
            writer.writeheader();writer.writerows(rows)
        result = subprocess.run([str(RSCRIPT),'--vanilla',str(driver),str(manifest),str(temp/'results.tsv'),str(temp/'runtime.txt')],
                                cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
        assert result.returncode==0,(result.stdout,result.stderr)
        with (temp/'results.tsv').open(encoding='utf-8',newline='') as inp:
            outcomes = list(csv.DictReader(inp,delimiter='\t'))
        assert len(outcomes)==63
        for outcome in outcomes:
            records[int(outcome.pop('id'))].update(outcome)
        runtime = (temp/'runtime.txt').read_text(encoding='utf-8').splitlines()
    # Ensure no source changed while the parser ran.
    for record,path in zip(records,files):
        assert record['sha256']==sha(path),f'Source changed during validation: {path}'
    receipt = {'status':'syntax_validation_complete','source_files':63,
               'Rscript_path':str(RSCRIPT),'Rscript_sha256':sha(RSCRIPT),'runtime':runtime,
               'R_driver_sha256':hashlib.sha256(R_DRIVER.encode()).hexdigest(),
               'analyses_executed':False,'production_writes':False,
               'prior_limitation_correction':'Rscript was unavailable on PATH, not absent. Installed R 4.6.1 was found and used for parse-only checks. No full analysis rerun is claimed.',
               'method':'R parse() on byte-identical temporary copies of R scripts and all fenced Rmd chunks, preserving source line positions; knitr::purl plus parse also checked when available. No analysis chunks executed. Knitr may evaluate chunk option expressions during purl; absent analysis-state names can emit warnings.',
               'passes':sum(record['status']=='pass' for record in records),
               'failures':sum(record['status']=='fail' for record in records),
               'purl_failures':sum(record['purl_status']=='fail' for record in records),
               'R_process_stdout':result.stdout,'R_process_stderr':result.stderr,'files':records,
               'limitations':['Syntax validation does not execute packages, joins, figures or tests and does not establish valid statistics/anatomy.',
                              'Inline prose R expressions and non-R code fences are outside the requested R-chunk parse check.']}
    path=HERE/args.receipt
    path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({key:receipt[key] for key in ['source_files','passes','failures','purl_failures','runtime']}))
