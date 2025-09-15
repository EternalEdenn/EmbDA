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

# Pipeline: Retrieval, Formatting, and Evaluation

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