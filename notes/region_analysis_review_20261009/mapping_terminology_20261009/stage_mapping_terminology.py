"""Stage wording only; never run producers/renderers or edit original artifacts."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source,old,new):
    if source.count(old)!=1:raise ValueError(f'Expected one occurrence: {old[:100]}')
    return source.replace(old,new,1)


def add_help(source,flag,help_text):
    lines=source.splitlines(keepends=True)
    found=[i for i,line in enumerate(lines) if f'parser.add_argument("--{flag}"' in line]
    if len(found)!=1:raise ValueError(f'CLI option not unique: {flag}')
    i=found[0];line=lines[i];j=line.rfind(')')
    lines[i]=line[:j]+', help='+repr(help_text)+line[j:]
    return ''.join(lines)


RENDER_LABELS={
 'mean template axon length density [mm/mm³/selected neuron]':'Axon-labelled length (mm/reference mm³/selected neuron)',
 'candidate endpoints/mm³/computable neuron':'Candidate axon ends/reference mm³/eligible neuron',
 'fraction of computable neurons with candidate endpoints':'Fraction of eligible neurons with ≥1 candidate axon end in the voxel',
 'mean of ':'equal mean of ',
 'selected=':'selected neurons=',
 '; computable=':'; eligible neurons=',
 'candidate graph endpoints; biological terminals unverified':'candidate axon ends; biological terminals unverified',
 'all-axon length, including arbors':'axon-labelled segment length; includes branches',
 'Matched anatomical MRI/map slices · NMT v2.1 symmetric\n':'Maps on population MRI slices · NMT v2.1 symmetric\n',
 'Shared MRI contrast/color scale; coordinate origin and anatomy provisional; no t test':'Descriptive values; registration and coordinate origin unverified',
}


class Normalize(ast.NodeTransformer):
    def visit_Module(self,node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body.pop(0)
        return node
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':node.keywords=[k for k in node.keywords if k.arg!='help']
        if isinstance(node.func,ast.Attribute) and node.func.attr=='suptitle':
            # Mask the title argument only. All calculations, call targets and fontsize remain compared.
            allowed=(ast.Constant,ast.JoinedStr,ast.FormattedValue,ast.IfExp,ast.Name,ast.Load)
            for n in ast.walk(node.args[0]):
                if not isinstance(n,allowed):raise ValueError(f'Non-display expression in suptitle: {type(n).__name__}')
                if isinstance(n,ast.Name) and n.id not in {'cuts','measurement','endpoint'}:raise ValueError('Unexpected title variable')
            node.args[0]=ast.Constant('__DISPLAY_TITLE__')
        return node
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            for before,after in RENDER_LABELS.items():
                if node.value==after:node.value=before
        return node


def main():
    paths=[ROOT/'group_analysis/scripts'/f for f in ('build_projection_maps.py','build_endpoint_maps.py','render_projection_slices.py')]
    paths += [ROOT/'group_analysis/evolution_20261008/figures'/d/'README.md' for d in ('multimonkey_coarse_20261009_layout-v2','additional_candidates_20261009')]
    protected=paths+[p for folder in ('projection_maps','endpoint_maps','figures') for p in (ROOT/'group_analysis/evolution_20261008'/folder).rglob('*') if p.is_file() and p.suffix.lower() in ('.gz','.png')]
    before={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in protected}
    stage=OUT/'staged';stage.mkdir(exist_ok=False)
    patch=[];validation=[]
    for path in paths:
        relative=path.relative_to(ROOT).as_posix();original=path.read_text(encoding='utf-8');updated=original
        if path.name=='build_projection_maps.py':
            updated=replace_once(updated,original.split('"""',2)[1],'''Build descriptive maps of axon-labelled segment length in NMT reference space.

Within each animal and selected source group, each voxel shows length summed
over selected neurons divided by their number. Density additionally divides
by reference voxel volume. Across animals, average the contributing animal
maps equally; a missing source group is not an observed zero. These are
descriptive physical measurements. Axon labels come from SWC node types and
still require compartment and registration review.

The manifest records AnimalID, SampleID, NeuronID, Subregion (the declared
source group), SWCPath, SWCSHA256, ReferenceSHA256, CoordinateFrame,
IndexScaleUm, AnatomyStatus and RegistrationStatus. atlas_index_um uses
three semicolon-separated scales; relative SWC paths use --input-root.
Exact identities and coordinate provenance must be supplied. The output
directory must be new. Source data and previous results remain preserved.
''')
            helps={'manifest':'CSV of exact neuron identities, source groups, SWC paths/hashes and coordinate provenance.','reference':'Pinned NMT reference image bound by every manifest row.','brain-mask':'Optional matching mask for coverage accounting; it does not trim the map.','input-root':'Base directory for relative SWC paths in the manifest.','space':'Existing reference-space name used in output filenames; this does not register data.','output':'New destination directory for descriptive length maps and provenance.'}
            for flag,h in helps.items():updated=add_help(updated,flag,h)
        elif path.name=='build_endpoint_maps.py':
            updated=replace_once(updated,original.split('"""',2)[1],'''Build descriptive maps of candidate axon ends and actual ARM-level labels.

A candidate axon end is a non-root axon-labelled (SWC type 2) node with no
children in the complete stored reconstruction. This graph rule does not
verify an image-reviewed terminal arbor, synapse or termination site.
An endpoint-eligible neuron has at least one such end anywhere, including
outside the reference field of view. Neurons without one remain unassessed.

For each animal and declared source group, endpoint count/density is the
candidate count per eligible neuron (density also divides by reference voxel
volume). Occupancy is the fraction of eligible neurons with at least one end
in a voxel; multiple ends from one neuron count once. Across animals, average
contributing animal maps equally. All selected neurons remain auditable;
no image review, registration or statistical inference occurs here.
''')
            old='parser.add_argument("--" + option, type=Path, required=True)'
            new='''parser.add_argument("--" + option, type=Path, required=True, help={
            "manifest": "CSV of exact neuron identities, source groups, hashes and coordinate provenance.",
            "reference": "Pinned reference image bound by the manifest.",
            "atlas-path": "Actual aligned ARM label volumes for direct voxel lookup.",
            "atlas-key": "Official ARM label names and indices.",
            "hemisphere-mask": "Matching mask with verified hemisphere label semantics.",
            "output": "New destination for candidate axon-end maps and audit records.",
        }[option])'''
            updated=replace_once(updated,old,new)
            for flag,h in {'brain-mask':'Optional coverage mask; endpoints inside the reference are retained even outside this mask.','input-root':'Base directory for relative SWC paths.','space':'Reference-space name for output filenames; no registration is performed.'}.items():updated=add_help(updated,flag,h)
        elif path.name=='render_projection_slices.py':
            updated=replace_once(updated,original.split('"""',2)[1],'''Display matched population MRI/map slices from an existing verified run.

Choose axon-labelled length density, candidate axon-end density, or the
fraction of endpoint-eligible neurons with an end in each voxel (occupancy).
Eligible means at least one non-root axon-labelled leaf anywhere in the
complete graph, including outside the reference field of view. Candidate
ends are not image-reviewed biological terminals.

Animal maps average within the selected source group. Group maps give equal
weight to contributing animals. Density color is log10(1 + density);
occupancy color is linear 0–1. Slice choices describe a display selection,
not full-volume anatomy or statistical significance. Source map values and
previous figures remain unchanged; the figure destination must be new.
''')
            for old,new in RENDER_LABELS.items():
                literal=repr(old)
                # Match source quoting without changing runtime identifiers.
                if old in ('mean of ','selected=','; computable='):
                    updated=updated.replace(old,new)
                else:
                    quoted=json.dumps(old,ensure_ascii=False)
                    updated=replace_once(updated,quoted,json.dumps(new,ensure_ascii=False))
            updated=replace_once(updated,'"Descriptive values; registration and coordinate origin unverified", fontsize=10)',
                '"Descriptive values; registration and coordinate origin unverified\\n"\n                        f"{\'Eligible: ≥1 non-root axon-labelled leaf anywhere (including outside view).\' if endpoint else \'Mean per selected neuron within each animal/source group.\'}", fontsize=10)')
            helps={'run':'Existing verified map run; source map values will not be changed.','readback':'Successful matching map-run review receipt.','output':'New destination for a separate figure variant.','metric':'Display axon-labelled length density, candidate axon-end density, or endpoint occupancy (eligible-neuron fraction).','scope':'Animal maps or the equal average of contributing animal maps.','cut-policy':'Choose separate cuts per map, shared data-selected cuts, or fixed common voxel indices.','slice-voxels':'Explicit common X Y Z voxel indices when --cut-policy=fixed; these are not millimetres.'}
            for flag,h in helps.items():updated=add_help(updated,flag,h)
        else:
            lines=updated.splitlines(keepends=True)
            lines.insert(2,'[Plain-language map terminology](../../../../docs/region_analysis_terminology.md). A candidate axon end is a non-root axon-labelled graph leaf, not a verified biological terminal. Endpoint-eligible neurons have at least one such leaf anywhere, including outside the reference field of view. Within each animal/source group, density is per eligible neuron (endpoints) or per selected neuron (axon length), and per reference mm³. Across animals, each contributing animal map has equal weight. Occupancy is the fraction of eligible neurons with at least one candidate end in a voxel, counting each neuron once.\n\n')
            updated=''.join(lines)
        destination=stage/relative;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(updated,encoding='utf-8',newline='\n')
        patch.extend(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),fromfile='a/'+relative,tofile='b/'+relative))
        result={'path':relative,'original_sha256':before[relative],'staged_sha256':sha(destination)}
        if path.suffix=='.py':
            compile(updated,str(destination),'exec')
            original_tree=Normalize().visit(ast.parse(original));staged_tree=Normalize().visit(ast.parse(updated))
            result['executable_AST_unchanged_except_explicit_display_text_and_CLI_help']=ast.dump(original_tree,include_attributes=False)==ast.dump(staged_tree,include_attributes=False)
            if not result['executable_AST_unchanged_except_explicit_display_text_and_CLI_help']:raise ValueError(f'Non-wording change detected: {relative}')
        validation.append(result)
    (OUT/'terminology.patch').write_text(''.join(patch),encoding='utf-8',newline='\n')
    changed=[p for p,h in before.items() if sha(ROOT/p)!=h]
    if changed:raise ValueError(f'Protected original changed: {changed}')
    report={'status':'static_wording_validation_passed','files':validation,'protected_original_hashes':before,'protected_original_count':len(before),'existing_maps_figures_or_production_files_changed':changed,'metric_IDs_columns_paths_formulas':'unchanged_by_AST_comparison','producers_and_renderer_executed':False,'figure_bytes_regenerated':False,'future_render_visual_validation':'required for a new figure variant; no claim of new layout acceptance','terminology_note_link':'root-owned docs/region_analysis_terminology.md','staging_script_sha256':sha(Path(__file__))}
    (OUT/'static_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'files':len(validation),'protected_originals':len(before),'status':report['status']}))


if __name__=='__main__':main()
