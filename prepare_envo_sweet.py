import csv
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from rdflib import Graph, OWL, RDF, RDFS

# 1. 设置 XML 文件路径（修改为你解压后的实际路径）
BIODIV_DIR = Path(r"D:\桌面\biodiv_2021owl\envo-sweet\component")
SRC_OWL = BIODIV_DIR / "source.xml"
TGT_OWL = BIODIV_DIR / "target.xml"
REF_XML = BIODIV_DIR / "reference.xml"
OUTPUT_CSV = Path("experiments/dataset/OAEI/Biodiv/small.csv")

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def get_classes_and_labels(owl_path):
  g = Graph()
  g.parse(str(owl_path), format="xml")
  classes = {}
  for s, _, o in g.triples((None, RDF.type, None)):
    if o in (OWL.Class, RDFS.Class):
      uri = str(s)
      label = ""
      for _, _, l in g.triples((s, RDFS.label, None)):
        label = str(l)
        break
      if not label:
        label = uri.split("#")[-1] if "#" in uri else uri.split("/")[-1]
      classes[uri] = label
  return classes


print("正在解析源本体和目标本体...")
src_classes = get_classes_and_labels(SRC_OWL)
tgt_classes = get_classes_and_labels(TGT_OWL)

# 2. 解析 reference.xml 获取标准参考对齐数据
print("正在解析 Reference 对齐映射...")
tree = ET.parse(REF_XML)
root = tree.getroot()
ns = {
    "align": "http://knowledgeweb.semanticweb.org/heterogeneity/alignment",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
}

ground_truth = {}
for cell in root.findall(".//align:Cell", ns):
  e1 = cell.find("align:entity1", ns).attrib.get(
      "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
  )
  e2 = cell.find("align:entity2", ns).attrib.get(
      "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
  )
  relation = cell.find("align:relation", ns).text
  if e1 in src_classes and e2 in tgt_classes and relation == "=":
    ground_truth[e1] = e2

# 3. 采样 20 个匹配对 + 20 个非匹配对（根据 KROMA/LLMs4OM 基准共生成 1000 对）
random.seed(42)
matched_src = list(ground_truth.keys())
unmatched_src = [c for c in src_classes.keys() if c not in ground_truth]
all_tgt_uris = list(tgt_classes.keys())

sampled_matched = random.sample(matched_src, min(20, len(matched_src)))
sampled_unmatched = random.sample(unmatched_src, min(20, len(unmatched_src)))

rows = []

# 有匹配的概念：每个概念包含 1 个正样本 + 24 个负样本
for s_uri in sampled_matched:
  t_true = ground_truth[s_uri]
  rows.append({
      "source": src_classes[s_uri],
      "source_uri": s_uri,
      "target": tgt_classes[t_true],
      "target_uri": t_true,
      "Relation": "1",
  })

  neg_targets = random.sample(
      [t for t in all_tgt_uris if t != t_true],
      min(24, len(all_tgt_uris) - 1),
  )
  for t_neg in neg_targets:
    rows.append({
        "source": src_classes[s_uri],
        "source_uri": s_uri,
        "target": tgt_classes[t_neg],
        "target_uri": t_neg,
        "Relation": "0",
    })

# 无匹配的概念：每个概念搭配 25 个负样本候选
for s_uri in sampled_unmatched:
  neg_targets = random.sample(all_tgt_uris, min(25, len(all_tgt_uris)))
  for t_neg in neg_targets:
    rows.append({
        "source": src_classes[s_uri],
        "source_uri": s_uri,
        "target": tgt_classes[t_neg],
        "target_uri": t_neg,
        "Relation": "0",
    })

# 4. 写入 CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
  writer = csv.DictWriter(
      f, fieldnames=["source", "source_uri", "target", "target_uri", "Relation"]
  )
  writer.writeheader()
  writer.writerows(rows)

print(f"完成！已生成包含 {len(rows)} 个采样样本对的文件：{OUTPUT_CSV}")