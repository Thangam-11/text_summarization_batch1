from mvp.mvp_summarizer import AbstractiveSummarizer, ExtractiveSummarizer

class SummarizationPipeline:
    def __init__(self, mode="abstractive", model="bart"):
        if mode == "abstractive":
            self.summarizer = AbstractiveSummarizer(model)
        else:
            self.summarizer = ExtractiveSummarizer()

    def run(self, text: str) -> str:
        return self.summarizer.summarize(text)
