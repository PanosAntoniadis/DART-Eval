import os
import sys
import wandb

from ....embeddings import RNALMEmbeddingExtractor
from ....components import SimpleSequence

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    model_name = "144M_H_MLM_last"
    use_track_embeddings = False
    genome_fa = os.path.join(root_output_dir,"refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    elements_tsv = os.path.join(root_output_dir,"task_3_peak_classification/processed_inputs/peaks_by_cell_label_unique_dataloader_format.tsv")
    chroms = None
    batch_size = 512
    num_workers = 4
    seed = 0
    device = "cuda"
    
    wandb.init(
        project="dart_eval_task3",
        name=f"embeddings_{model_name}_trackEmbeddings{use_track_embeddings}",
        entity="RNALM",
        dir="outputs/wandb",
        config={
            "model_name": model_name,
            "use_track_embeddings": use_track_embeddings,
            "task": "task_3",
            "approach": "extract_embeddings",
            "batch_size": batch_size,
            "num_workers": num_workers,
            "seed": seed,
            "device": device,
            "chroms": chroms,
        }
    )

    if use_track_embeddings is True:
        out_path = os.path.join(root_output_dir,f"task_3_peak_classification/embeddings/{model_name}_trackEmbeddings.h5")
    else:
        out_path = os.path.join(root_output_dir,f"task_3_peak_classification/embeddings/{model_name}.h5")

    dataset = SimpleSequence(genome_fa, elements_tsv, chroms, seed)
    extractor = RNALMEmbeddingExtractor(model_name, use_track_embeddings, batch_size, num_workers, device)
    extractor.extract_embeddings(dataset, out_path, progress_bar=True)

    wandb.finish()
