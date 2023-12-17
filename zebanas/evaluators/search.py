
class SearcherEvaluator:
    def __init__(self, searcher, score_evaluator):
        self.searcher = searcher
        self.score_evaluator = score_evaluator

    def __call__(
        self,
        cfg,
        algorithm,
        dataloader,
        chromosomes,
        latency
    ):
        chromosomes = self.searcher.search(
            cfg,
            algorithm,
            dataloader,
            chromosomes,
            latency
        )

        if chromosomes is None:
            return 0.

        score = self.score_evaluator.get_single(
            cfg,
            chromosomes,
            dataloader,
            device=cfg.execute.device
        )
        return score
