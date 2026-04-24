import argparse
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import pprint
import json
from scipy import stats

pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser()
   
# configuration.
parser.add_argument('--anno', type=str, default="capt/annotation.txt")
parser.add_argument('--ignored_phones', type=str, default="<eps>")
parser.add_argument('--capt_dir', type=str, default="exp/models/tdnn/decode/capt")

args = parser.parse_args()

sent_anno = {}
sent_pred = {}

spk2l1 = {  "ABA":"Arabic", "SKA":"Arabic", "YBAA":"Arabic", "ZHAA":"Arabic", 
            "BWC":"Mandarin", "LXC":"Mandarin", "NCC":"Mandarin", "TXHC":"Mandarin", 
            "ASI":"Hindi", "RRBI":"Hindi", "SVBI":"Hindi", "TNI":"Hindi", "HJK":"Korean", 
            "HKK":"Korean", "YDCK":"Korean", "YKWK":"Korean", "EBVS":"Spanish", 
            "ERMS":"Spanish", "MBMPS":"Spanish", "NJS":"Spanish", "HQTV":"Vietnamese", 
            "PNV":"Vietnamese", "THV":"Vietnamese","TLV":"Vietnamese"}

def process_fn(info_fn, sent_dict):
    #print("process_fn:", info_fn)
    with open(info_fn, "r") as fn:
        for line1, line2, line3, line4 in itertools.zip_longest(*[fn]*4):
            ref_info = line1.split()
            utt_id = ref_info[0]
            
            ref_info = ref_info[2:]
            hyp_info = line2.split()[2:]
            op_info = line3.split()[2:]
            # ignore 4rd line (non-sense)
            assert len(ref_info) == len(hyp_info) and len(hyp_info) == len(op_info)
            
            sent_dict[utt_id] = {"ref":[], "hyp":[], "op":[]}
            # extract alignment information from the file
            for i in range(len(ref_info)):
                #if op_info[i] == "I": continue
                sent_dict[utt_id]["ref"].append(ref_info[i])
                sent_dict[utt_id]["hyp"].append(hyp_info[i])
                sent_dict[utt_id]["op"].append(op_info[i])
    return sent_dict

def comp_metric(sent_anno, sent_pred, l1=None):
    #print("comp_metric:")
    # Detection
    anno_list = []
    pred_list = []
    # Diagnose
    diag_report = {"ce":0, "de":0, "de_ori": []}
    utt_ids = list(sent_anno.keys())
    # some details
    anno_hyp_list = []
    pred_hyp_list = []
    
    for utt_id in utt_ids:
        anno_info = sent_anno[utt_id]
        pred_info = sent_pred[utt_id]
        spk = utt_id.split("_")[0]
        spk_l1 = spk2l1[spk]
        
        if l1 is not None:
            if spk_l1 != l1:
                continue
       
        assert len(anno_info["op"]) == len(pred_info["op"])
        
        for i in range(len(anno_info["op"])):
            # annotation
            if anno_info["op"][i] == "C":
                anno = 1
            else:
                anno = 0
            # prediction
            if pred_info["op"][i] == "C":
                pred = 1
            else:
                pred = 0
            anno_list.append(anno)
            pred_list.append(pred)
            # diagose
            if pred == 0 and anno == 0:
                if anno_info["hyp"][i] == pred_info["hyp"][i]:                
                    diag_report["ce"] += 1
                else:
                    diag_report["de"] += 1
                    diag_report["de_ori"].append([anno_info["hyp"][i],  pred_info["hyp"][i]])
            # details
            anno_hyp_list.append(anno_info["hyp"][i])
            pred_hyp_list.append(pred_info["hyp"][i])
            
    return anno_list, pred_list, diag_report, anno_hyp_list, pred_hyp_list

#def report_error_details(sent_anno, sent_pred):
def report_error_details(sent_anno):
    # return (MP_info):
    # { "l1_id": {"err_type": [correct_detect, correct_diagnose, total]}}
    utt_ids = list(sent_anno.keys())
    l1_stats = {}
    
    for utt_id in utt_ids:
        anno_info = sent_anno[utt_id]
        #pred_info = sent_pred[utt_id]
        #assert len(anno_info["op"]) == len(pred_info["op"])
        
        spk = utt_id.split("_")[0]
        l1 = spk2l1[spk]
        
        if l1 not in l1_stats:
            l1_stats[l1] = {}
        
        for i in range(len(anno_info["op"])):
            # MDD analysis
            err_type = anno_info["op"][i]
            
            if err_type == "S":
                err_info = "S_" + anno_info["ref"][i] + "->" + anno_info["hyp"][i] 
            elif err_type == "D":
                err_info = "D_" + anno_info["ref"][i] + "->SIL"
            elif err_type == "I":
                err_info = "I_SIL->" + anno_info["hyp"][i]
            else:
                continue
            
            # [correct detect, correct diagnose, total]
            if err_info not in l1_stats[l1]:
                l1_stats[l1][err_info] = [0, 0, 0]
                
            # correct detect
            #if pred_info["op"][i] in ["S", "D"]:
            #    l1_stats[l1][err_info][0] += 1
                
            # correct diagnose
            #if pred_info["hyp"][i] == anno_info["hyp"][i]:
            #    l1_stats[l1][err_info][1] += 1
                
            l1_stats[l1][err_info][2] += 1
                
    return l1_stats    

#def report_accuracy_details(sent_anno, sent_pred):
def report_accuracy_details(sent_anno):
    # return (MP_info):
    # { "l1_id": {"err_type": [correct_detect, correct_diagnose, total]}}
    utt_ids = list(sent_anno.keys())
    l1_stats = {}
    
    for utt_id in utt_ids:
        anno_info = sent_anno[utt_id]
        #pred_info = sent_pred[utt_id]
        #assert len(anno_info["op"]) == len(pred_info["op"])
        
        spk = utt_id.split("_")[0]
        l1 = spk2l1[spk]
        
        if l1 not in l1_stats:
            l1_stats[l1] = {}
        
        for i in range(len(anno_info["op"])):
            # MDD analysis
            err_type = anno_info["op"][i]
            
            if err_type == "S":
                err_info = "S_" + anno_info["ref"][i] + "->" + anno_info["hyp"][i] 
            elif err_type == "C":
                err_info = "C_" + anno_info["ref"][i] + "->" + anno_info["hyp"][i]
            elif err_type == "D":
                err_info = "D_" + anno_info["ref"][i] + "->SIL"
            elif err_type == "I":
                err_info = "I_SIL->" + anno_info["hyp"][i]
            else:
                raise ValueError()          
            
            # [correct detect, correct diagnose, total]
            if err_info not in l1_stats[l1]:
                l1_stats[l1][err_info] = [0, 0, 0]
                
            # correct detect
            #if err_type in ["S", "D"] and pred_info["op"][i] in ["S", "D"]:
            #    l1_stats[l1][err_info][0] += 1
            #elif err_type in ["C"] and pred_info["op"][i] in ["C"]:
            #    l1_stats[l1][err_info][0] += 1
                
            # correct diagnose
            #if pred_info["hyp"][i] == anno_info["hyp"][i]:
            #    l1_stats[l1][err_info][1] += 1
                
            l1_stats[l1][err_info][2] += 1
                
    return l1_stats

def output_csv_info(capt_dir, l1_stats, csv_affix = ""):
    err_types = {}
    
    # speaker-wise information
    with open(capt_dir + "/per_l1_anno_only"+csv_affix+".csv", "w") as fn:
        fn.write("l1,err_type,cdetect,cdiagnose,total\n")
        for l1_id in list(l1_stats.keys()):
            l1_err_types = l1_stats[l1_id]
            for err_type in list(l1_err_types.keys()):
                info = l1_id + "," + err_type
                err_stats = l1_err_types[err_type]
                
                if err_type not in err_types:
                    err_types[err_type] = [0, 0, 0]
                
                for i in range(len(err_types[err_type])):
                    err_types[err_type][i] += err_stats[i]
                    info += "," + str(err_stats[i])
                fn.write(info + "\n")
    
    # corpus-wise information
    with open(capt_dir + "/per_all_anno_only"+csv_affix+".csv", "w") as fn:
        fn.write("err_type,cdetect,cdiagnose,total\n")
        for err_type in list(err_types.keys()):
            info = err_type
            err_stats = err_types[err_type]
                
            for i in range(len(err_types[err_type])):
                info += "," + str(err_stats[i])
            fn.write(info + "\n")

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

if __name__ == "__main__":
    sent_anno = process_fn(args.anno, sent_anno)
    #sent_pred = process_fn(args.pred, sent_pred)
    #anno_list, pred_list, diag_report, anno_hyp_list, pred_hyp_list = comp_metric(sent_anno, sent_pred)
    
    #tn, fp, fn, tp = confusion_matrix(anno_list, pred_list).ravel()
    #print("tn, fp, fn, tp")
    #print(tn, fp, fn, tp)
    #pp.pprint(classification_report(anno_list, pred_list, output_dict=True))
    #print(diag_report["ce"], diag_report["de"], diag_report["ce"] / (diag_report["ce"] + diag_report["de"]))
    
    #print(anno_list, score_list) 
    #print(stats.spearmanr(anno_list, score_list))
    #print()
    #l1_list = list(set(spk2l1.values()))
    #for l1 in l1_list:
    #    #print("L1", l1)
    #    anno_list, pred_list, diag_report, anno_hyp_list, pred_hyp_list = comp_metric(sent_anno, sent_pred, l1)
    #    #pp.pprint(classification_report(anno_list, pred_list, output_dict=True))
    #print()
    
    # print(diag_report["de_ori"])
    ignored_phones = args.ignored_phones.split(",") 
    phone_dict = {"<unk>": 0}
    
    # plot_phn_conf_mat(anno_hyp_list, pred_hyp_list, phone_dict, args.capt_dir)
    l1_stats = report_error_details(sent_anno)
    l1_acc_stats = report_accuracy_details(sent_anno)
    
    output_csv_info(args.capt_dir, l1_stats)
    output_csv_info(args.capt_dir, l1_acc_stats, "_acc")
