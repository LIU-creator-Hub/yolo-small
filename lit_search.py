import json, time, urllib.parse, urllib.request
QUERIES = {
 "gap": [
   "partial deformable convolution",
   "sparse deformable convolution",
   "deformable convolution background suppression steel surface defect",
   "octave convolution crack detection",
   "high frequency guided feature aggregation strip steel",
   "multi branch strip dilated depthwise convolution attention",
   "learnable morphology convolution neural network",
 ],
}
BASE = "https://api.openalex.org/works"; MAIL = "lit-search@example.com"
def fetch(q, n=8):
    url = f"{BASE}?search={urllib.parse.quote(q)}&per_page={n}&mailto={MAIL}"
    req = urllib.request.Request(url, headers={"User-Agent": "lit-search/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r: return json.load(r)
out = {}
for q in QUERIES["gap"]:
    try:
        d = fetch(q); items=[]
        for w in d.get("results", []):
            loc = (w.get("primary_location") or {}).get("source") or {}
            inv = w.get("abstract_inverted_index"); abs_txt=""
            if inv:
                pos={}
                for word,idxs in inv.items():
                    for i in idxs: pos[i]=word
                abs_txt=" ".join(pos[k] for k in sorted(pos))
            items.append({"title":w.get("title"),"year":w.get("publication_year"),
                          "venue":loc.get("display_name"),"doi":w.get("doi"),
                          "cited":w.get("cited_by_count"),"abstract":abs_txt})
        out[q]=items; print(f"{q} -> {len(items)}")
    except Exception as e:
        out[q]=[]; print(f"{q} -> ERROR {e}")
    time.sleep(0.3)
json.dump(out, open(r"G:\deeplearning\YOLO-small\lit_search_gap.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
with open(r"G:\deeplearning\YOLO-small\lit_gap_report.txt","w",encoding="utf-8") as f:
    for q, items in out.items():
        f.write("="*90+"\nQ: "+q+"\n")
        for i,it in enumerate(items,1):
            f.write(f"{i}. [{it.get('year')}] cites={it.get('cited')} | {it.get('title')}\n   {it.get('venue')} | {it.get('doi')}\n")
            ab=(it.get('abstract') or '').replace('\n',' ')
            if ab: f.write("   abstract: "+ab[:600]+"\n")
print("saved gap files")
