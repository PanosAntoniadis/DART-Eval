import os
import sys
import wandb

from ....embeddings import RNALMEmbeddingExtractor
from ....components import SimpleSequence

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    model_name = "rnalm_144M_HM_MM"
    output_dir = "/tmp/vqj407/rnalm_erda/gefion_output//outputs/mlm_track_metadata/runs/2025-06-29_21-09-26_144M_human_mouse_resume/"
    checkpoint_path = "best"
    use_metadata = False
    tokenizer_path = "/home/vqj407/workspace/RNALM/rnalm/tokenizers/dna_tokenizer"
      
    genome_fa = os.path.join(root_output_dir,"refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    cell_line = sys.argv[1] #cell line name
    category = sys.argv[2] #peaks, nonpeaks, or idr
    if category == "idr":
        elements_tsv = os.path.join(root_output_dir, f"task_4_chromatin_activity/processed_data/cell_line_idr_peaks/{cell_line}.bed")
    else:
        elements_tsv = os.path.join(root_output_dir, f"task_4_chromatin_activity/processed_data/cell_line_expanded_peaks/{cell_line}_{category}.bed")
    chroms = None
    batch_size = 64
    num_workers = 4
    seed = 0
    device = "cuda"
    
    wandb.init(
        project="dart_eval_task4",
        name=f"embeddings_{model_name}_{cell_line}_{category}",
        entity="RNALM",
        dir="outputs/wandb",
        config={
            "model_name": model_name,
            "checkpoint_path": checkpoint_path,
            "output_dir": output_dir,
            "use_metadata": use_metadata,
            "tokenizer_path": tokenizer_path,
            "cell_line": cell_line,
            "category": category,
            "task": "task_4",
            "approach": "extract_embeddings",
            "batch_size": batch_size,
            "num_workers": num_workers,
            "seed": seed,
            "device": device,
            "chroms": chroms,
        }
    )

    out_dir = os.path.join(root_output_dir, f"task_4_chromatin_activity/embeddings/{model_name}/")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cell_line}_{category}.h5")
      
    dataset = SimpleSequence(genome_fa, elements_tsv, chroms, seed)
    extractor = RNALMEmbeddingExtractor(output_dir, checkpoint_path, use_metadata,
                                        tokenizer_path, batch_size, num_workers, device)
    extractor.extract_embeddings(dataset, out_path, progress_bar=True)

    wandb.finish()
