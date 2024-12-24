import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dataset import WordVocab
import argparse
import pickle

from utils.pMHC_TCR_pred_attn import binding_predict_single, binding_predict_pmhc_specific

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", "--mode", type=str, default="single", help="single or batch prediction")
    # single
    parser.add_argument("-alphav", "--alphav", type=str, default=None, help="alpha V gene")
    parser.add_argument("-alphaj", "--alphaj", type=str, default=None, help="alpha J gene")
    parser.add_argument("-betav", "--betav", type=str, default=None, help="beta V gene")
    parser.add_argument("-betaj", "--betaj", type=str, default=None, help="beta J gene")
    parser.add_argument("-alpha_cdr3", "--alpha_cdr3", type=str, default=None, help="alpha CDR3")
    parser.add_argument("-beta_cdr3", "--beta_cdr3", type=str, default=None, help="beta CDR3")
    parser.add_argument("-epitope", "--epitope", type=str, default=None, help="epitope sequence")
    parser.add_argument("-mhc", "--mhc", type=str, default=None, help="MHC allele")
    parser.add_argument("-predict_interaction", "--predict_interaction", required=False, type=str, default="True", 
                        help="predict interaction between bCDR3 and epitope or not")
    # batch
    parser.add_argument("-data_path", "--data_path", type=str, default=None, help="path to the data file")
    parser.add_argument("-o", "--output_path", type=str, default="./binding_res.csv", 
                        help="path to the output file, default is ./binding_res.csv")
    # universal
    parser.add_argument("-batch_size", "--batch_size", type=int, default=32, help="batch size for model prediction")
    parser.add_argument("-cuda", "--with_cuda", required=False, type=str, default="True", help="use cuda or not")
    parser.add_argument("-is_cal_rank", "--is_cal_rank", required=False, type=str, default="True", help="calculate rank score or not")
    parser.add_argument("-seed", "--seed", type=int, default=3407, help="random seed for reproducibility")
    parser.add_argument("-bg_num", "--bg_num", type=int, default=1000, help="number of background TCRs")
    parser.add_argument("-healthy_bg", "--healthy_bg", type=str, default="False", help="use healthy background TCRs (True) or all background TCRs (False)")
    
    args = parser.parse_args()
    # 将字符串转换为bool
    args.healthy_bg = args.healthy_bg.lower() == "true"
    args.with_cuda = args.with_cuda.lower() == "true"
    args.is_cal_rank = args.is_cal_rank.lower() == "true"
    args.predict_interaction = args.predict_interaction.lower() == "true"
    
    with open("TcrDesign/tmp/args.pkl", "wb") as f:
        pickle.dump(args, f)
        
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    if args.mode == "single":
        binding_score, rank_score, attn_weight = binding_predict_single(args.alphav, args.alphaj, args.alpha_cdr3, 
                                                                        args.betav, args.betaj, args.beta_cdr3, 
                                                                        args.epitope, args.mhc, 
                                                                        args.with_cuda, args.is_cal_rank, args.seed, args.bg_num, args.batch_size)
        if args.predict_interaction:
            # 画图输出
            matrix = attn_weight
            cmap = LinearSegmentedColormap.from_list("red_gradient", ["#fff1e3", "#c2302c"])
            fig, ax = plt.subplots()
            im = ax.imshow(matrix, cmap=cmap, interpolation='nearest')
            ax.set_frame_on(False)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.outline.set_visible(False)
            ax.set_xticks(list(range(0, matrix.shape[1])), [aa for aa in args.epitope])
            ax.set_yticks(list(range(0, matrix.shape[0])), [aa for aa in args.beta_cdr3])
            plt.savefig('./interaction_plot.pdf', format='pdf', bbox_inches='tight')
        
        # Output the results
        print("Binding score: ", binding_score, "\nRank score (%): ", rank_score)
        
    elif args.mode == "batch":
        binding_df = binding_predict_pmhc_specific(args.data_path, args.batch_size, args.with_cuda, False, args.is_cal_rank, args.seed, args.bg_num)
        binding_df.to_csv(args.output_path, index=False)
    else:
        raise ValueError("Invalid mode, please choose 'single' or 'batch'")
    
    print("Prediction completed")
    
    os.remove("TcrDesign/tmp/args.pkl")
