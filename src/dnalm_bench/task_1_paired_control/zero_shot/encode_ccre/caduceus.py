import os
import wandb

from ..evaluators import PairedControlDataset, CaduceusEvaluator

os.environ["TOKENIZERS_PARALLELISM"] = "false"

work_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    model_name = "caduceus-ps_seqlen-131k_d_model-256_n_layer-16"

    genome_fa = os.path.join(work_dir, "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    elements_tsv = os.path.join(work_dir, f"task_1_ccre/processed_inputs/ENCFF420VPZ_processed.tsv")

    out_dir = os.path.join(work_dir, f"task_1_ccre/zero_shot_outputs/likelihoods/{model_name}")

    chroms = [
        "chr5",
        "chr10",
        "chr14",
        "chr18",
        "chr20",
        "chr22"
    ]

    batch_size = 512
    num_workers = 4
    seed = 0
    device = "cuda"

    # Initialize wandb
    wandb.init(
        project="dart_eval_task1",
        name="zero_shot_caduceus",
        entity="RNALM",
        dir="outputs/wandb",
        config={
            "model_name": model_name,
            "task": "task_1",
            "approach": "zero_shot",
            "batch_size": batch_size,
            "num_workers": num_workers,
            "seed": seed,
            "device": device,
            "chroms": chroms,
        }
    )

    dataset = PairedControlDataset(genome_fa, elements_tsv, chroms, seed)
    evaluator = CaduceusEvaluator(model_name, dataset, batch_size, num_workers, device)
    metrics = evaluator.evaluate(out_dir, progress_bar=True)

    for k, v in metrics.items():
        print(f"{k}: {v}")

    wandb.log(metrics)

    wandb.finish()
