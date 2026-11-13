import json
from pathlib import Path
from rdflib import OWL, RDF, RDFS, Graph, URIRef

# 1. 原始本体文件路径（请替换为你电脑上的实际绝对路径）
BIODIV_DIR = Path(r"D:\biodiv_2021owl\envo-sweet\component")
SRC_OWL = BIODIV_DIR / "source.xml"  # ENVO
TGT_OWL = BIODIV_DIR / "target.xml"  # SWEET

OUT_DIR = Path("experiments/dataset/OAEI/Biodiv")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IAO_DEFINITION = URIRef("http://purl.obolibrary.org/obo/IAO_0000115")
EXACT_SYNONYM = URIRef(
    "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"
)
RELATED_SYNONYM = URIRef(
    "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym"
)


def build_dict_jsonl(owl_path: Path, output_jsonl: Path):
  print(f"正在解析本体: {owl_path} ...")
  g = Graph()
  g.parse(str(owl_path), format="xml")

  classes = set()
  for s, _, o in g.triples((None, RDF.type, None)):
    if o in (OWL.Class, RDFS.Class) and isinstance(s, URIRef):
      classes.add(s)

  records = {}
  for cls_uri in classes:
    uri_str = str(cls_uri)

    # 提取标签 Label
    labels = []
    for _, _, lbl in g.triples((cls_uri, RDFS.label, None)):
      labels.append(str(lbl))
    if not labels:
      fallback = (
          uri_str.split("#")[-1] if "#" in uri_str else uri_str.split("/")[-1]
      )
      labels.append(fallback)

    # 提取父类 Parents
    parents = []
    for _, _, parent in g.triples((cls_uri, RDFS.subClassOf, None)):
      if isinstance(parent, URIRef):
        p_uri = str(parent)
        p_lbl = ""
        for _, _, pl in g.triples((parent, RDFS.label, None)):
          p_lbl = str(pl)
          break
        if not p_lbl:
          p_lbl = (
              p_uri.split("#")[-1] if "#" in p_uri else p_uri.split("/")[-1]
          )
        parents.append(p_lbl)

    # 提取同义词 Synonyms
    synonyms = []
    for pred in (EXACT_SYNONYM, RELATED_SYNONYM):
      for _, _, syn in g.triples((cls_uri, pred, None)):
        synonyms.append(str(syn))

    # 提取定义 Definition / Comment
    definition = ""
    for _, _, d in g.triples((cls_uri, IAO_DEFINITION, None)):
      definition = str(d)
      break
    if not definition:
      for _, _, c in g.triples((cls_uri, RDFS.comment, None)):
        definition = str(c)
        break

    records[uri_str] = {
        "labels": labels,
        "parents": parents,
        "synonyms": synonyms,
        "definition": definition,
    }

  print(f"写入词典缓存 -> {output_jsonl} (共 {len(records)} 个条目)...")
  with open(output_jsonl, "w", encoding="utf-8") as f:
    for uri_str, meta in records.items():
      item = {
          "uri": uri_str,
          "labels": meta["labels"],
          "parents": meta["parents"],
          "synonyms": meta["synonyms"],
          "definition": meta["definition"],
      }
      f.write(json.dumps(item, ensure_ascii=False) + "\n")


# 2. 执行生成 ENVO 和 SWEET 两个词典
build_dict_jsonl(SRC_OWL, OUT_DIR / "envo_dict.jsonl")
build_dict_jsonl(TGT_OWL, OUT_DIR / "sweet_dict.jsonl")
print("两个词典缓存均已生成完毕！")