import os
import sys
import wandb

from ....embeddings import RNALMEmbeddingExtractor
from ....components import SimpleSequence

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    model_name = "rnalm_144M_H_MLM"
    output_dir = "/tmp/vqj407/rnalm_erda/gefion_output//outputs/mlm_pretraining/runs/2025-03-26_00-14-15_evenLargerModel/"
    checkpoint_path = "best"
    use_metadata = True
    tokenizer_path = "/home/vqj407/workspace/RNALM/rnalm/tokenizers/dna_tokenizer"
    genome_fa = os.path.join(root_output_dir,"refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    elements_tsv = os.path.join(root_output_dir,"task_3_peak_classification/processed_inputs/peaks_by_cell_label_unique_dataloader_format.tsv")
    chroms = None
    batch_size = 512
    num_workers = 0
    seed = 0
    device = "cuda"
    
    wandb.init(
        project="dart_eval_task3",
        name=f"embeddings_{model_name}",
        entity="RNALM",
        dir="outputs/wandb",
        config={
            "model_name": model_name,
            "checkpoint_path": checkpoint_path,
            "output_dir": output_dir,
            "use_metadata": use_metadata,
            "tokenizer_path": tokenizer_path,
            "task": "task_3",
            "approach": "extract_embeddings",
            "batch_size": batch_size,
            "num_workers": num_workers,
            "seed": seed,
            "device": device,
            "chroms": chroms,
        }
    )

    out_path = os.path.join(root_output_dir,f"task_3_peak_classification/embeddings/{model_name}_{use_metadata}.h5")

    dataset = SimpleSequence(genome_fa, elements_tsv, chroms, seed)
    extractor = RNALMEmbeddingExtractor(output_dir, checkpoint_path, use_metadata,
                                        tokenizer_path, batch_size, num_workers, device)
    extractor.extract_embeddings(dataset, out_path, progress_bar=True)

    wandb.finish()
