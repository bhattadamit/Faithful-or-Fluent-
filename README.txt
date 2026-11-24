Online Resource 1

Article: Faithful or Fluent? An Evidence-Calibrated Screening Model for Evaluating Retrieval-Augmented Answers on US Consumer-Banking Procedures
Journal: Knowledge and Information Systems
Authors: 
Corresponding author: 

Contents
1. Faithful_or_Fluent_replication_analysis.py reproduces every number in Tables 5-12 and Figs. 1-7.
   Run it from this folder: python Faithful_or_Fluent_replication_analysis.py (needs numpy, scipy, matplotlib).
   It reads the re-analysis outputs in reanalysis/ and writes results and figures to out/.
2. reanalysis/ holds the aggregated evaluator outputs used in Section 6.7 (Tables 11 and 12, Fig. 7).
3. Faithful_or_Fluent_reanalysis_kit.zip regenerates the reanalysis/ files from the released RAGTruth,
   HaluBench and FaithBench data by scoring every item with HHEM-2.1-Open and the number check (see its README).
