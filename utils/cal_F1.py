import argparse 
from utils import*

def main():
    parser = argparse.ArgumentParser(description='Parameters')

    parser.add_argument('--data_domain', type=str, default='Army', help='choose from Army, Hiru, ITN, Newsfirst')
    parser.add_argument('--da_method', type=str, default='mean', help='document alignment method, choose from [mean, tkpert, mean-ot, mean-tkpert, mean-maxsim, tkpert-maxsim, mean-bimax, tkpert-bimax]')
    parser.add_argument('--sim_method', type=str, default='cos', help='choose from [cos, margin_score]')
    parser.add_argument('--lang_pair', type=str, default='en-si', help='choose from [en-si, en-ta, si-ta]')
    parser.add_argument('--split_method', type=str, default='ofls', help='choose from [ofls, sbs]')
    parser.add_argument('--overlap', type=float, default= 0.5, help='(The rate of overlap)')
    parser.add_argument('--seg_len', type=int, default=30, help='lenth of segment for sliding window')
    parser.add_argument('--data_path', type=str, default='../fernando_data', help='Path to data')
    parser.add_argument('--out_path', type=str, default='../da_results', help='path to output')
    args = parser.parse_args()

    lang_pair = args.lang_pair
    da_method = args.da_method
    domain = args.data_domain
    retri = args.sim_method
    if args.split_method == "ofls":
        split_md = f"fl{args.seg_len}_or{args.overlap}"
    else:
        split_md = "sbs"

    pred_pairs = load_pairs(args.out_path + f"/{domain}/{domain}_{lang_pair}_{split_md}.{retri}.{da_method}_result.txt")
    gold_pairs = load_pairs(args.data_path + f"/{domain}/{domain.lower()}_{lang_pair}_golden-pair.txt")

    tp = len(pred_pairs & gold_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print(f"Recall: {recall*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"F1: {f1*100:.2f}%")

if __name__ == '__main__':
    main()
