import pathlib

raw = pathlib.Path('data/raw/iirs')
print('Subdirs:', [d.name for d in raw.iterdir() if d.is_dir()])
for d in raw.iterdir():
    if not d.is_dir():
        continue
    xmls = list(d.rglob('*.xml'))
    qubs = list(d.rglob('*.qub'))
    print(f"\n{d.name}: {len(xmls)} xmls, {len(qubs)} qubs")
    for x in xmls:
        sibling_qub = x.with_suffix('.qub').exists()
        is_browse = 'browse' in str(x).lower()
        print(f"  xml={x.relative_to(raw)} | has_qub={sibling_qub} | browse={is_browse}")
