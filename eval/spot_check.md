# Manual spot-check of the LLM judge (10 answers)

Sample: the 4 lowest-scored GraphRAG answers, 3 random GraphRAG answers and 3 random vanilla answers
(seed 7), each read against its reference and the cited label passages.

| id | judge score | my score | why |
|---|---|---|---|
| graphrag:m7 | 0.5 | 1 | States CYP2D6 inhibition by fluoxetine, tramadol metabolism via CYP2D6, seizure + serotonin-syndrome risk, "monitor" verdict; only omits the words "less analgesia" |
| graphrag:c2 | 0.5 | 1 | Quotes the label's exact criteria (serum creatinine ≥1.5 mg/dL males, ≥1.4 females, abnormal CrCl); omits "lactic acidosis" |
| graphrag:m1 | 1 | 1 | Contraindicated; CYP3A4 inhibition → higher simvastatin levels → myopathy/rhabdomyolysis |
| graphrag:m2 | 1 | 1 | Contraindicated; CYP3A4 mechanism; myopathy/rhabdomyolysis |
| graphrag:s4 | 1 | 1 | Hyperkalemia risk; monitor serum potassium |
| graphrag:m5 | 1 | 1 | Avoid; CYP2C19 needed to activate clopidogrel; reduced antiplatelet effect |
| graphrag:m10 | 1 | 1 | Digoxin levels up ~70% via P-gp inhibition; measure levels / adjust dose |
| vanilla:m2 | 1 | 1 | Contraindicated; CYP3A4; myopathy |
| vanilla:m3 | 0.5 | 0.5 | Reports the prothrombin-time rise but not the CYP2C9 mechanism or INR monitoring |
| vanilla:c2 | 1 | 1 | Exact label criteria |

**Agreement: 8/10.** Both disagreements are cases where the judge was stricter than the reference warrants,
and both are GraphRAG answers — so the reported GraphRAG accuracy is, if anything, slightly conservative.

Also observed: the judge is not fully deterministic across batches (identical vanilla answers scored 95% in one
run and 90% in the next), so accuracy differences of about ±5 points between pipelines are within noise.
