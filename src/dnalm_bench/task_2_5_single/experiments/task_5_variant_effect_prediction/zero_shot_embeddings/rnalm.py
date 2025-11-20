import os
import sys
import numpy as np
import polars as pl
import wandb

from ....evaluators import RNALMVariantEmbeddingEvaluator
from ....components import VariantDataset

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    dataset = sys.argv[1]
    model_name = "rnalm_144M_HM_MM"
    use_track_embeddings = False
    batch_size = 512
    num_workers = 0
    seed = 0
    device = "cuda"
    chroms=None

    variants_bed = sys.argv[1]
    output_prefix = sys.argv[2]
    genome_fa = sys.argv[3]
    cell_line = "GM12878"

    out_dir = os.path.join(root_output_dir, f"task_5_variant_effect_prediction/outputs/zero_shot/embeddings/{model_name}")
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, output_prefix + ".tsv")

    allele1_embeddings_path = os.path.join(out_dir, f"{output_prefix}_allele1_embeddings.npy")
    allele2_embeddings_path = os.path.join(out_dir, f"{output_prefix}_allele2_embeddings.npy")

    wandb.init(
        project="dart_eval_task5",
        name=f"zs_embedding_{model_name}_{output_prefix}",
        entity="RNALM",
        dir="outputs/wandb",
        config={
            "model_name": model_name,
            "task": "task_5",
            "approach": "zero_shot_embeddings",
            "batch_size": batch_size,
            "num_workers": num_workers,
            "seed": seed,
            "device": device,
        }
    )
    dataset = VariantDataset(genome_fa, variants_bed, chroms, seed)
    evaluator = RNALMVariantEmbeddingEvaluator(model_name, use_track_embeddings, batch_size, num_workers, device)
    score_df, allele1_embeddings, allele2_embeddings = evaluator.evaluate(dataset, out_path, progress_bar=True)

    df = dataset.elements_df
    scored_df = pl.concat([df, score_df], how="horizontal")
    print(out_path)
    scored_df.write_csv(out_path, separator="\t")


    # Log scored table
    scored_pd = scored_df.to_pandas()
    wandb.log({"scored_table": wandb.Table(dataframe=scored_pd)})

    wandb.finish()