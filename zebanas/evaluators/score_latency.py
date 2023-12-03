class ScoreAndLatencyEvaluator:
    def __init__(
        self,
        score_evaluator,
        latency_evaluator
    ):
        self.score_evaluator = score_evaluator
        self.latency_evaluator = latency_evaluator

    def __call__(
        self,
        cfg,
        samples,
        dataloader,
        chromosomes,
        search_index
    ):
        scores = self.score_evaluator(
            cfg=cfg,
            samples=samples,
            dataloader=dataloader,
            chromosomes=chromosomes,
            search_index=search_index
        )
        latencies = self.latency_evaluator(cfg, samples, search_index)

        objs = [[s, l] for s, l in zip(scores, latencies)]

        return samples.set_obj(objs)
