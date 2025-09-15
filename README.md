# BiMax: Bidirectional MaxSim Score for Document-Level Alignment
This **EmbDA** repository contains all the document alignment methods we used in the paper.
- **Mean-Pool**
- **TK-PERT**
- **OT (Optimal Transport)**: Part of the code refers to [OTAlign](https://aclanthology.org/2023.acl-long.219/).
- **GMD (Greedy Movers' Distance)**: Part of the code refers to [Fernando et al.](https://github.com/nlpcuom/parallel_corpus_mining/blob/master/document_alignment/GreedyMoversDistance.py).
- **BiMax**

# Prerequisite
- Environment: See `environment.yml`.
- Dataset: We list the [Fernando Dataset](https://github.com/kdissa/comparable-corpus) used in the paper in our github repository.
  - The dataset is placed in `fernando_data/` directory.
  - If you'd like to use this dataset in your research, please cite their [paper](https://link.springer.com/article/10.1007/s10115-022-01761-x).

# Generate Embeddings
**Quick Start**: Run `main/00.generate_embs.sh` to generate embeddings.
``` shell
cd ./main
sh 00.generate_embs.sh
``` 
Modify the parameters `LANG`, `DATA_DOMAIN`, and `SPLIT_METHOD` in the shell file to generate embeddings using the **OFLS** or **SBS** for different languages across the various domains.
- Note: When setting `DATA_DOMAIN`, please pay attention to case sensitivity.
``` shell
#!/bin/bash

LANG=en                    # Lang, choose from [en, si, ta]
DATA_DOMAIN=Army      # Data domain, choose from [Newsfirst, ITN, Army, Hiru]

TKPERT_J=16                # J for TK-PERT
TKPER_SHAPE=20             # Shape for TK-PERT

SPLIT_METHOD=ofls          # Split method, choose from [sbs, ofls]
# if ofls has been chosen
FL=30                      # Fixed-Length for ofls
OVERLAP=0.5                # Overlapping Rate for ofls

DATA_PATH=../fernando_data
OUT_PATH=../embs             # Path to save the embeddings
mkdir -p $OUT_PATH

CHUNK_SIZE=2048              # Chunk size used to split document segments into batches for embedding
SAVE_NUM=4096                # Document Num. for saving in one embedding file


echo "Begin generating embeddings..."
python ../utils/generate_embs.py    --seg_len $FL --overlap $OVERLAP --split_method $SPLIT_METHOD \
                                    --J $TKPERT_J --shape $TKPER_SHAPE \
                                    --lang $LANG \
                                    --data_domain $DATA_DOMAIN \
                                    --data_path $DATA_PATH \
                                    --out_path $OUT_PATH \
                                    --chunk_size $CHUNK_SIZE \
                                    --save_num $SAVE_NUM
``` 

# Pipeline: Retrieval, Formatting, and Evaluation
`main/01.pipeline.sh` is a pipeline that integrates retrieval, result formatting, and evaluation. Individual components can be used separately if needed.
``` shell
sh 01.pipeline.sh
``` 
To use this pipeline, please configure the following:
- Source language `SRC_LANG`
- Target language `TGT_LANG`
- Data domain `DATA_DOMAIN`
- Segmentation strategy `SPLIT_METHOD`
- Document alignment method `DA_METHOD`
  - Choose `[mean, tkpert]` to only use Mean-Pool or TK-PERT
  - Choose `mean-[sf, sl]-[ot, gmd]` to use Mean-Pool vectors as retrieval embeddings, with:
    - "sf" or "sl" as weighting scheme
    - OT or GMD for alignment
  - Choose `mean-bimax` to use BiMax
- Similarity computation method when using Mean-Pool or TK-PERT `SIM_METHOD`
  - Please choose from "cos" and "margin"
- Search Strategy using FAISS `SEARCH_TYPE`
- Number of candidates for FAISS search `CAND_NUM`

The default settings correspond to the parameters used in the paper.
``` shell
#!/bin/bash

EMBS_PATH=../embs
OUT_PATH=../da_results
DATA_PATH=../fernando_data

mkdir -p $OUT_PATH

SRC_LANG=si               # Source Language
TGT_LANG=ta               # Target Language

DA_METHOD=mean-sf-ot    
# Document alignment method, choose from [mean, tkpert, mean-[sl, sf]-[ot, gmd] (e.g., mean-sf-ot), mean-bimax]

SIM_METHOD=cos             # Retrieval strategy for "mean" or "tkpert", choose from [cos, margin]
SPLIT_METHOD=ofls          # Segmentation method, choose from [ofls, sbs]
DATA_DOMAIN=ITN            # Data domain, choose from [Newsfirst, ITN, Army, Hiru]
# If "ofls" is choosed
FL=30                      # Fixed-Length for OFLS
OR=0.5                     # Overlapping Rate for OFLS

CAND_NUM=32                # Maximum candidate number for each source doucment using faiss search
SEARCH_TYPE=cos            # Search type using faiss, choose from [cos, L2]

LANG_PAIR=${SRC_LANG}-${TGT_LANG}

...
``` 
# Citation
If you find our paper and code helpful in your research, please cite our paper:

``` shell
@inproceedings{wang-etal-2024-document,
    title = "Document Alignment based on Overlapping Fixed-Length Segments",
    author = "Wang, Xiaotian  and
      Utsuro, Takehito  and
      Nagata, Masaaki",
    editor = "Fu, Xiyan  and
      Fleisig, Eve",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-srw.10/",
    doi = "10.18653/v1/2024.acl-srw.10",
    pages = "51--61",
    ISBN = "979-8-89176-097-4",
    abstract = "Acquiring large-scale parallel corpora is crucial for NLP tasks such as Neural Machine Translation, and web crawling has become a popular methodology for this purpose. Previous studies have been conducted based on sentence-based segmentation (SBS) when aligning documents in various languages which are obtained through web crawling. Among them, the TK-PERT method (Thompson and Koehn, 2020) achieved state-of-the-art results and addressed the boilerplate text in web crawling data well through a down-weighting approach. However, there remains a problem with how to handle long-text encoding better. Thus, we introduce the strategy of Overlapping Fixed-Length Segmentation (OFLS) in place of SBS, and observe a pronounced enhancement when performing the same approach for document alignment. In this paper, we compare the SBS and OFLS using three previous methods, Mean-Pool, TK-PERT (Thompson and Koehn, 2020), and Optimal Transport (Clark et al., 2019; El-Kishky and Guzman, 2020), on the WMT16 document alignment shared task for French-English, as well as on our self-established Japanese-English dataset MnRN. As a result, for the WMT16 task, various SBS based methods showed an increase in recall by 1{\%} to 10{\%} after reproduction with OFLS. For MnRN data, OFLS demonstrated notable accuracy improvements and exhibited faster document embedding speed."
}
``` 