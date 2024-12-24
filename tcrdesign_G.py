import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dataset import WordVocab
from utils import generate_betaTCRs, generate_alphaTCRs, generate_VJ

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-epitope", "--epitope", required=False, type=str, help="epitope sequence for TCR generation")
    parser.add_argument("-num", "--gen_num", required=False, type=int, default=1000, help="number of TCRs to generate")
    parser.add_argument("-maxLen", "--max_len", required=False, type=int, default=20, help="maximum length of TCRs")
    parser.add_argument("-cuda", "--with_cuda", required=False, type=str, default="True", help="use cuda for TCR generation")
    parser.add_argument("-t", "--temperature", required=False, type=float, default=1, help="temperature for TCR generation")
    parser.add_argument("-o", "--output_path", required=False, type=str, default="./generate_res.csv", 
                        help="output path for generate TCRs, default is ./generate_res.csv")
    
    parser.add_argument("-mode", "--mode", required=False, type=str, default="beta", 
                        help="select mode for TCR generation: beta, alpha or vj, default is beta")
    parser.add_argument("-bcdr3", "--bcdr3", required=False, type=str, default=None, help="bcdr3 for aTCR generation / bcdr3 for vj generation")
    parser.add_argument("-acdr3", "--acdr3", required=False, type=str, default=None, help="acdr3 for vj generation")
    parser.add_argument("-gen_num_vj", "--gen_num_vj", required=False, type=int, default=5, help="number of VJs to generate")
    
    args = parser.parse_args()
    # 将字符串转换为bool
    args.with_cuda = args.with_cuda.lower() == "true"
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    if args.mode == "beta":
        decoder_res_list = generate_betaTCRs(args.epitope, args.gen_num, args.max_len, args.with_cuda, args.temperature)
        df = pd.DataFrame({"TCRs": decoder_res_list})
        df.to_csv(args.output_path, index=False)
    elif args.mode == "alpha":
        if args.bcdr3 is None:
            raise ValueError("bcdr3 is required for alpha TCR generation")
        decoder_res_list = generate_alphaTCRs(args.epitope, args.bcdr3, args.gen_num, args.max_len, args.with_cuda, args.temperature)
        df = pd.DataFrame({"TCRs": decoder_res_list})
        df.to_csv(args.output_path, index=False)
    elif args.mode == "vj":
        if args.bcdr3 is not None:
            decoder_res_list = generate_VJ(args.bcdr3, args.gen_num_vj, args.with_cuda, is_alpha=False, batch_mode=False)
            df = pd.DataFrame({"V": [vj_pair[0] for vj_pair in decoder_res_list], "J": [vj_pair[1] for vj_pair in decoder_res_list]})
            df.to_csv(args.output_path, index=False)
        elif args.acdr3 is not None:
            decoder_res_list = generate_VJ(args.acdr3, args.gen_num_vj, args.with_cuda, is_alpha=True, batch_mode=False)
            df = pd.DataFrame({"V": [vj_pair[0] for vj_pair in decoder_res_list], "J": [vj_pair[1] for vj_pair in decoder_res_list]})
            df.to_csv(args.output_path, index=False)
        else:
            raise ValueError("Either bcdr3 or acdr3 is required for vj generation")
    else:
        raise ValueError("Invalid mode: " + args.mode)
    
    print("Generation completed")    
