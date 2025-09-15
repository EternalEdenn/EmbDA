#!/bin/bash

EMBS_PATH=../embs
OUT_PATH=../da_results
DATA_PATH=../fernando_data

mkdir -p $OUT_PATH

SRC_LANG=en               # Source Language
TGT_LANG=si               # Target Language

DA_METHOD=mean-bimax    
# Document alignment method, choose from [mean, tkpert, mean-[sl, sf]-[ot, gmd] (e.g., mean-sf-ot), mean-bimax]

# If "bimax" is choosed
BIMAX_TYPE=loop            # Calculation type of BiMax, choose from [loop, batch] 

SIM_METHOD=cos             # Retrieval strategy for "mean" or "tkpert", choose from [cos, margin]
SPLIT_METHOD=ofls          # Segmentation method, choose from [ofls, sbs]
DATA_DOMAIN=Army            # Data domain, choose from [Newsfirst, ITN, Army, Hiru]
# If "ofls" is choosed
FL=30                      # Fixed-Length for OFLS
OR=0.5                     # Overlapping Rate for OFLS

CAND_NUM=32                # Maximum candidate number for each source doucment using faiss search
SEARCH_TYPE=cos            # Search type using faiss, choose from [cos, L2]

LANG_PAIR=${SRC_LANG}-${TGT_LANG}


python ../utils/retrieval.py    --data_domain $DATA_DOMAIN \
                                --embs_path $EMBS_PATH \
                                --out_path $OUT_PATH \
                                --da_method $DA_METHOD \
                                --bimax_type $BIMAX_TYPE \
                                --sim_method $SIM_METHOD \
                                --src_lang $SRC_LANG \
                                --tgt_lang $TGT_LANG \
                                --split_method $SPLIT_METHOD \
                                --seg_len $FL \
                                --overlap $OR \
                                --faiss_k $CAND_NUM \
                                --faiss_search $SEARCH_TYPE

python ../utils/turn2result.py  --data_domain $DATA_DOMAIN \
                                --out_path $OUT_PATH \
                                --lang_pair $LANG_PAIR \
                                --split_method $SPLIT_METHOD \
                                --seg_len $FL \
                                --overlap $OR \
                                --sim_method $SIM_METHOD \
                                --da_method $DA_METHOD

python ../utils/cal_F1.py   --data_domain $DATA_DOMAIN \
                            --out_path $OUT_PATH \
                            --data_path $DATA_PATH \
                            --lang_pair $LANG_PAIR \
                            --split_method $SPLIT_METHOD \
                            --seg_len $FL \
                            --overlap $OR \
                            --sim_method $SIM_METHOD \
                            --da_method $DA_METHOD

