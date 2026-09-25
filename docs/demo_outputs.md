# Example answers (live, 2026-09-25)

Question: Can a patient taking simvastatin start clarithromycin?

## graphrag
```
Contraindicated [clarithromycin:contraindications:2].

Clarithromycin is a strong inhibitor of the CYP3A4 enzyme [clarithromycin:drug_interactions:0], which is responsible for metabolizing simvastatin [simvastatin:drug_interactions:2]. Concomitant administration leads to significantly increased plasma concentrations of simvastatin [clarithromycin:warnings_and_cautions:8]. This elevation drastically increases the risk of myopathy and rhabdomyolysis [clarithromycin:contraindications:2].

--- provider: gemini/gemini-3.5-flash-lite (disk cache) | cache hit: False | timings: {'retrieve_ms': 53, 'generate_ms': 7975, 'total_ms': 8027}
--- entities: ['clarithromycin', 'simvastatin'] | path nodes: ['amiodarone', 'amlodipine', 'clarithromycin', 'cyclosporine', 'cyp3a4', 'diltiazem', 'rifampin', 'simvastatin', 'verapamil']
--- citations: ['clarithromycin:contraindications:2', 'clarithromycin:drug_interactions:0', 'simvastatin:drug_interactions:2', 'clarithromycin:warnings_and_cautions:8']
```

## vanilla
```
Contraindicated.

Simvastatin is extensively metabolized by the CYP3A4 enzyme, and concomitant treatment with clarithromycin increases its plasma concentration [clarithromycin:warnings_and_cautions:8]. This elevated statin exposure significantly increases the risk of myopathy, including rhabdomyolysis [clarithromycin:warnings_and_cautions:8]. Therefore, the simultaneous use of clarithromycin and simvastatin is contraindicated, and simvastatin therapy must be suspended during the course of clarithromycin treatment if it cannot be avoided [clarithromycin:warnings_and_cautions:8, clarithromycin:contraindications:2].

--- provider: gemini/gemini-3.5-flash-lite (disk cache) | cache hit: False | timings: {'retrieve_ms': 10, 'generate_ms': 9103, 'total_ms': 9113}
--- entities: [] | path nodes: []
--- citations: ['clarithromycin:warnings_and_cautions:8', 'clarithromycin:contraindications:2']
```
