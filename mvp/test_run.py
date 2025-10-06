from mvp.nlp_models import SummarizerAPI, ParaphraserAPI

summarizer = SummarizerAPI()
paraphraser = ParaphraserAPI()

text = "Artificial Intelligence is transforming industries by enabling automation and decision-making."

print("Summary:", summarizer.summarize(text))
print("Paraphrase:", paraphraser.paraphrase(text))
