import os
import sys

from ....embeddings import RNALMEmbeddingExtractor
from ....components import SimpleSequence

root_output_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    model_name = "rnalm"
    output_dir = "/tmp/vqj407/rnalm_erda/gefion_output/outputs/mlm_track_metadata/runs/2025-07-15_11-11-12_144M_fukushima_w_taxonomy_no_tax_loss_resume"
    checkpoint_path = "best"
    use_metadata = False
    tokenizer_path = "/home/vqj407/workspace/RNALM/rnalm/tokenizers/dna_tokenizer"
    genome_fa = os.path.join(root_output_dir,"refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    elements_tsv = os.path.join(root_output_dir,"task_3_peak_classification/processed_inputs/peaks_by_cell_label_unique_dataloader_format.tsv")
    chroms = None
    batch_size = 64
    num_workers = 0
    seed = 0
    device = "cuda"

    out_path = os.path.join(root_output_dir,f"task_3_peak_classification/embeddings/{model_name}.h5")

    dataset = SimpleSequence(genome_fa, elements_tsv, chroms, seed)
    extractor = RNALMEmbeddingExtractor(output_dir, checkpoint_path, use_metadata,
                                        tokenizer_path, batch_size, num_workers, device)
    extractor.extract_embeddings(dataset, out_path, progress_bar=True)
