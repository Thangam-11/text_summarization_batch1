from mvp_pipeline import SummarizationPipeline

text = """
Artificial Intelligence is transforming industries worldwide. 
From healthcare to finance, it automates processes and provides insights. 
Machine learning models help doctors diagnose diseases, 
while banks use AI to detect fraud. 
Education also benefits, with AI-driven tutoring systems personalizing learning.
"""

print("\n--- Abstractive (BART) ---")
bart = SummarizationPipeline(mode="abstractive", model="bart")
print(bart.run(text))

print("\n--- Abstractive (T5) ---")
t5 = SummarizationPipeline(mode="abstractive", model="t5")
print(t5.run(text))

print("\n--- Extractive (TextRank) ---")
extractive = SummarizationPipeline(mode="extractive")
print(extractive.run(text))
