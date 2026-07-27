# run_insula_flatmap.jl — L2 orchestrator for insula / multi-monkey flatmaps
#
# Thin CLI over existing L1 entrypoints (no Gou/monkeyrec rewrite):
#   single    → run_insula_pipeline          (gou_flatmap_minimal.jl)
#   multi     → run_multi_monkey_flatmap     (multi_monkey_flatmap.jl)
#   lr_mirror → run_insula_lr_mirror         (insula_lr_mirror_flatmap.jl)
#
# Usage:
#   julia +1.11.5 --project=<monkeyrec> run_insula_flatmap.jl --help
#   julia +1.11.5 --project=<monkeyrec> run_insula_flatmap.jl --mode single --dry-run
#   julia +1.11.5 --project=<monkeyrec> run_insula_flatmap.jl --mode multi --force
#
# Prefer the PowerShell launcher (pins Julia 1.11.5 + JULIA_CPU_THREADS=1):
#   .\run_insula_flatmap.ps1 -Mode single -DryRun

using Dates
using XLSX

const SCRIPT_DIR = @__DIR__
const PROJECT_ROOT = dirname(SCRIPT_DIR)

# ── Defaults (match current working paths) ─────────────────────────
const DEFAULT_MONKEYREC = joinpath(PROJECT_ROOT, "references",
                                   "analysis-code_gou_etal_2025", "monkeyrec")
const DEFAULT_ATLAS     = joinpath(PROJECT_ROOT, "atlas", "NMT_v2.0_sym")
const DEFAULT_SOMA_SINGLE = joinpath(PROJECT_ROOT, "neuron_tables",
                                     "251637_INS_HE_inferred.xlsx")
const DEFAULT_SOMA_MULTI  = joinpath(PROJECT_ROOT, "group_analysis", "combined",
                                     "multi_monkey_INS_combined.xlsx")
const DEFAULT_OUT_SINGLE  = joinpath(PROJECT_ROOT, "figures_charts",
                                     "gou_flatmap_conservative")
const DEFAULT_OUT_MULTI   = joinpath(PROJECT_ROOT, "group_analysis", "R_analysis",
                                     "outputs", "figures", "flatmap")
const DEFAULT_CACHE       = joinpath(DEFAULT_OUT_SINGLE, "cache")
const DEFAULT_NITER       = 30000

const MODE_OUTPUT_KEYS = Dict(
    "single" => ["insula_base.png",
                 "insula_soma_LR_combined.png",
                 "insula_soma_LR_split.png",
                 "insula_soma_type.png"],
    "multi" => ["flatmap_all_monkeys_combined.png",
                "flatmap_per_monkey_panels.png"],
    "lr_mirror" => ["insula_LR_mirror_flatmaps.png"],
)

# ── CLI ────────────────────────────────────────────────────────────
function print_help()
    println("""
    run_insula_flatmap.jl — insula / multi-monkey flatmap orchestrator

    Usage: run_insula_flatmap.ps1 -Mode <single|multi|lr_mirror> [-Sheet <name>] [options]

    Primary inputs:
      --mode VAL         single | multi | lr_mirror            (required)
      --sheet NAME       sheet name in soma xlsx                (default: auto — uses Summary if present,
                         otherwise prompts interactively; pass --sheet <name> to skip prompt)

    The sheet contains all data (SampleID, Soma_Side, Soma_Region, Neuron_Type,
    Soma_Phys_X/Y/Z). Samples, sides, regions, types, colors, panel grid, and
    titles are auto-discovered from the sheet — no flags for those.

    Optional (sensible defaults):
      --soma PATH        soma xlsx path                         (default: auto by mode)
                         single: neuron_tables/251637_INS_HE_inferred.xlsx
                         multi:  group_analysis/combined/multi_monkey_INS_combined.xlsx
      --out PATH         output dir                            (default: auto by mode)
                         single/lr: figures_charts/gou_flatmap_conservative
                         multi:     group_analysis/R_analysis/outputs/figures/flatmap
      --cache PATH       JLD2 cache dir                        (default: .../gou_flatmap_conservative/cache)
      --niter N          flatten iterations                    (default: $(DEFAULT_NITER))
      --force            overwrite existing outputs            (default: off)
      --dated            write to run_YYYYMMDD_HHMMSS/ subdir   (default: off)
      --dry-run          validate paths + print params; no plot (default: off)
      --monkeyrec PATH   Gou monkeyrec project root             (default: references/analysis-code_gou_etal_2025/monkeyrec)
      --atlas PATH       NMT atlas root (informational)        (default: atlas/NMT_v2.0_sym)
      --help, -h         this message

    Auto-inferred from sheet (no flag):
      samples (sort(unique(SampleID))), sides (L+R, R mirrored), regions, types,
      title (sample IDs + counts), panel grid (ceil(sqrt(N))),
      label (true for single, false for multi), markersize=8, px_per_unit=3, png+svg

    Examples:
      .\\run_insula_flatmap.ps1 -Mode multi
      .\\run_insula_flatmap.ps1 -Mode multi -Sheet Summary
      .\\run_insula_flatmap.ps1 -Mode single -Dated
      .\\run_insula_flatmap.ps1 -Mode multi -Soma D:\\path\\to\\other_table.xlsx -Sheet Neurons

    Defaults match gou_flatmap_minimal.jl / multi_monkey_flatmap.jl.
    Multi-monkey figure drift vs older plots is expected when the combined
    soma table grows — do not require pixel-identical reproduction.
    """)
end

function _take_value!(args, i, flag)
    i < length(args) || error("Missing value after $flag")
    return args[i + 1], i + 1
end

function parse_cli(args)
    cfg = Dict{Symbol,Any}(
        :mode => nothing,
        :dry_run => false,
        :force => false,
        :dated => false,
        :help => false,
        :niter => DEFAULT_NITER,
        :soma => nothing,
        :sheet => nothing,
        :out => nothing,
        :cache => DEFAULT_CACHE,
        :atlas => DEFAULT_ATLAS,
        :monkeyrec => DEFAULT_MONKEYREC,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("--help", "-h")
            cfg[:help] = true
        elseif a == "--dry-run"
            cfg[:dry_run] = true
        elseif a == "--force"
            cfg[:force] = true
        elseif a == "--dated"
            cfg[:dated] = true
        elseif a == "--mode"
            v, i = _take_value!(args, i, a)
            cfg[:mode] = lowercase(strip(v))
        elseif a == "--niter"
            v, i = _take_value!(args, i, a)
            cfg[:niter] = parse(Int, v)
        elseif a == "--soma"
            v, i = _take_value!(args, i, a)
            cfg[:soma] = v
        elseif a == "--sheet"
            v, i = _take_value!(args, i, a)
            cfg[:sheet] = v
        elseif a == "--out"
            v, i = _take_value!(args, i, a)
            cfg[:out] = v
        elseif a == "--cache"
            v, i = _take_value!(args, i, a)
            cfg[:cache] = v
        elseif a == "--atlas"
            v, i = _take_value!(args, i, a)
            cfg[:atlas] = v
        elseif a == "--monkeyrec"
            v, i = _take_value!(args, i, a)
            cfg[:monkeyrec] = v
        else
            error("Unknown argument: $a (try --help)")
        end
        i += 1
    end
    return cfg
end

default_soma(mode::AbstractString) =
    mode == "multi" ? DEFAULT_SOMA_MULTI : DEFAULT_SOMA_SINGLE

default_out(mode::AbstractString) =
    mode == "multi" ? DEFAULT_OUT_MULTI : DEFAULT_OUT_SINGLE

"""Resolve figure output directory; apply --dated stamp when requested."""
function resolve_figure_dir(mode::AbstractString, out_root::AbstractString; dated::Bool)
    base = mode == "multi" ? out_root : joinpath(out_root, "insula")
    if dated
        stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        return joinpath(base, "run_$stamp")
    end
    return base
end

function existing_outputs(mode::AbstractString, fig_dir::AbstractString)
    keys = get(MODE_OUTPUT_KEYS, mode, String[])
    return filter(isfile, [joinpath(fig_dir, k) for k in keys])
end

function cache_status(cache_dir::AbstractString, niter::Int)
    depth = joinpath(cache_dir, "depth_volume.jld2")
    flat  = joinpath(cache_dir, "flatmap_leftinsula_n$(niter).jld2")
    return (
        depth_path = depth,
        flat_path = flat,
        depth_hit = isfile(depth),
        flat_hit = isfile(flat),
    )
end

function print_run_summary(cfg; fig_dir::AbstractString)
    cs = cache_status(cfg[:cache], cfg[:niter])
    println("="^60)
    println("Insula flatmap run summary")
    println("="^60)
    println("  mode          : $(cfg[:mode])")
    println("  dry_run       : $(cfg[:dry_run])")
    println("  force         : $(cfg[:force])")
    println("  dated         : $(cfg[:dated])")
    println("  niter         : $(cfg[:niter])")
    println("  soma          : $(cfg[:soma])  exists=$(isfile(cfg[:soma]))")
    println("  sheet         : $(cfg[:sheet])")
    println("  atlas         : $(cfg[:atlas])  exists=$(isdir(cfg[:atlas]))")
    println("  monkeyrec     : $(cfg[:monkeyrec])  exists=$(isdir(cfg[:monkeyrec]))")
    println("  cache         : $(cfg[:cache])")
    println("  depth cache   : $(cs.depth_hit ? "HIT" : "MISS")  $(cs.depth_path)")
    println("  flatmap cache : $(cs.flat_hit ? "HIT" : "MISS")  $(cs.flat_path)")
    println("  figure dir    : $fig_dir")
    coords = joinpath(cfg[:monkeyrec], "monkeytemp", "wyz-upload", "julia",
                      "flatmap-brainarea-coords.jl")
    coords_alt = joinpath(SCRIPT_DIR, "monkeytemp", "wyz-upload", "julia",
                          "flatmap-brainarea-coords.jl")
    println("  coords (cwd)  : $coords  exists=$(isfile(coords))")
    println("  coords (alt)  : $coords_alt  exists=$(isfile(coords_alt))")
    println("="^60)
end

function validate_cfg!(cfg)
    mode = cfg[:mode]
    mode in ("single", "multi", "lr_mirror") ||
        error("--mode must be single|multi|lr_mirror (got $(repr(mode)))")
    cfg[:force] && cfg[:dated] &&
        error("Use either --force or --dated, not both")
    cfg[:soma] === nothing && (cfg[:soma] = default_soma(mode))
    cfg[:out]  === nothing && (cfg[:out]  = default_out(mode))
    isfile(cfg[:soma]) || error("Soma table not found: $(cfg[:soma])")
    isdir(cfg[:atlas]) || error("Atlas root not found: $(cfg[:atlas])")
    isdir(cfg[:monkeyrec]) || error("monkeyrec root not found: $(cfg[:monkeyrec])")
    coords = joinpath(cfg[:monkeyrec], "monkeytemp", "wyz-upload", "julia",
                      "flatmap-brainarea-coords.jl")
    isfile(coords) || @warn "Missing flatmap-brainarea-coords.jl under monkeyrec cwd" path=coords
    return cfg
end

"""
    resolve_sheet!(cfg; dry_run) -> cfg

Resolve the xlsx sheet to use and store it in `cfg[:sheet]`.

- If `--sheet <name>` was passed (cfg[:sheet] !== nothing): use it directly;
  error if not present in the workbook.
- Else (auto): if `Summary` exists, use it silently; if only one sheet exists,
  use it silently; otherwise list sheets and prompt by number (in dry-run,
  list and pick the first without blocking).
"""
function resolve_sheet!(cfg; dry_run::Bool)
    path = cfg[:soma]
    override = cfg[:sheet]
    sheets = XLSX.sheetnames(XLSX.readxlsx(path))
    if override !== nothing
        s = String(override)
        if s in sheets
            cfg[:sheet] = s
            @info "Sheet selected (explicit)" sheet=s available=sheets
        else
            error("Sheet '$s' not found in $path. Available sheets: $(join(sheets, ", "))")
        end
        return cfg
    end
    # auto
    if "Summary" in sheets
        cfg[:sheet] = "Summary"
        @info "Sheet auto-selected" sheet="Summary" available=sheets
    elseif length(sheets) == 1
        cfg[:sheet] = sheets[1]
        @info "Sheet auto-selected (only sheet)" sheet=sheets[1] available=sheets
    else
        println("Multiple sheets found in $path:")
        for (i, s) in enumerate(sheets)
            println("  $i. $s")
        end
        if dry_run
            println("Would prompt for sheet (pass -Sheet <name> to skip prompt). Dry-run: not blocking.")
            cfg[:sheet] = sheets[1]
        else
            print("Pick sheet [1-$(length(sheets))] (or pass -Sheet <name> to skip prompt): ")
            line = readline()
            idx = tryparse(Int, strip(line))
            if idx === nothing || !(1 <= idx <= length(sheets))
                error("Invalid sheet selection: '$line' (expected 1-$(length(sheets)))")
            end
            cfg[:sheet] = sheets[idx]
            @info "Sheet selected (prompt)" sheet=sheets[idx]
        end
    end
    cfg
end

"""Print a prominent banner showing the resolved soma file + sheet."""
function print_soma_banner(cfg)
    println("━"^3, " Soma table ", "━"^3)
    println("  file:   $(cfg[:soma])")
    println("  sheet:  $(cfg[:sheet])")
    println("  exists: $(isfile(cfg[:soma]) ? "yes" : "no")")
    println("━"^17)
end

function guard_overwrite!(mode::AbstractString, fig_dir::AbstractString; force::Bool, dated::Bool)
    dated && return
    hits = existing_outputs(mode, fig_dir)
    isempty(hits) && return
    if force
        @info "Overwriting existing outputs (--force)" files=hits
        return
    end
    error(string(
        "Output files already exist in $fig_dir\n",
        "  ", join(basename.(hits), ", "), "\n",
        "Re-run with --force to overwrite, or --dated for a stamped subfolder."))
end

# ── Modes ──────────────────────────────────────────────────────────
function run_dry!(cfg)
    fig_dir = resolve_figure_dir(cfg[:mode], cfg[:out]; dated=cfg[:dated])
    print_run_summary(cfg; fig_dir)
    hits = existing_outputs(cfg[:mode], fig_dir)
    if !isempty(hits) && !cfg[:force] && !cfg[:dated]
        println("  NOTE: existing outputs would block a real run (use --force or --dated)")
        for h in hits
            println("    - $h")
        end
    else
        println("  overwrite check: OK")
    end
    cs = cache_status(cfg[:cache], cfg[:niter])
    if cfg[:mode] == "multi" && !(cs.depth_hit && cs.flat_hit)
        println("  NOTE: multi mode requires cache HIT for depth + flatmap")
    end
    println("Dry-run complete (no plotting).")
    return 0
end

function run_live!(cfg)
    fig_dir = resolve_figure_dir(cfg[:mode], cfg[:out]; dated=cfg[:dated])
    guard_overwrite!(cfg[:mode], fig_dir; force=cfg[:force], dated=cfg[:dated])
    mkpath(fig_dir)
    print_run_summary(cfg; fig_dir)

    # zz2fig cwd-relative includes (flatmap-brainarea-coords.jl)
    cd(cfg[:monkeyrec])

    mode = cfg[:mode]
    niter = cfg[:niter]
    soma = cfg[:soma]
    sheet = cfg[:sheet]
    cache = cfg[:cache]

    # invokelatest: L1 methods are defined by include() after run_live! was
    # compiled — avoid Julia world-age MethodError on the first call.
    if mode == "single"
        include(joinpath(SCRIPT_DIR, "gou_flatmap_minimal.jl"))
        Base.invokelatest(run_insula_pipeline; niter, soma_table=soma, sheet=sheet,
                          out_dir=fig_dir, cache_dir=cache)
    elseif mode == "multi"
        include(joinpath(SCRIPT_DIR, "multi_monkey_flatmap.jl"))
        Base.invokelatest(run_multi_monkey_flatmap; niter, soma_table=soma, sheet=sheet,
                          out_dir=fig_dir, cache_dir=cache)
    elseif mode == "lr_mirror"
        include(joinpath(SCRIPT_DIR, "insula_lr_mirror_flatmap.jl"))
        Base.invokelatest(run_insula_lr_mirror; niter, soma_table=soma, sheet=sheet,
                          out_dir=fig_dir, cache_dir=cache)
    end
    println("Done. Figures under: $fig_dir")
    return 0
end

function main(args=ARGS)
    cfg = parse_cli(args)
    if cfg[:help]
        print_help()
        return 0
    end
    if cfg[:mode] === nothing
        print_help()
        println("ERROR: Missing --mode (single|multi|lr_mirror)")
        return 1
    end
    validate_cfg!(cfg)
    resolve_sheet!(cfg; dry_run=cfg[:dry_run])
    print_soma_banner(cfg)
    return cfg[:dry_run] ? run_dry!(cfg) : run_live!(cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(main())
    catch e
        if e isa ErrorException
            println(stderr, "ERROR: ", e.msg)
            exit(1)
        else
            rethrow()
        end
    end
end
