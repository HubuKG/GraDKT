import os
from time import time
from pprint import pformat

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import BCELoss, CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F
from torch.optim import Adam

from dataset import KTDataset, pad_collate
from util import MainArgParser, set_seed, write_log, set_config, load_model, timestamp, calc_metric, result_to_csv
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(args):
    device = args.device
    args.device = device
    max_patience = 20

    # 使用单一的数据文件，不做交叉验证
    train_files = ["train.json"]
    valid_files = ["test.json"]
    train_set = KTDataset(args.dataset_path, train_files, args.max_concept_len)
    valid_set = KTDataset(args.dataset_path, valid_files, args.max_concept_len)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, collate_fn=pad_collate)
    valid_loader = DataLoader(valid_set, batch_size=args.batch, shuffle=False, collate_fn=pad_collate)

    model = load_model(args.model, args).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    kt_loss = BCELoss()
    ot_loss = CrossEntropyLoss(ignore_index=-1)
    pos_weight = torch.tensor((args.concept - args.top_k) / args.top_k, device=args.device)
    r_loss = BCEWithLogitsLoss(pos_weight=pos_weight)
    cl_loss = CrossEntropyLoss(ignore_index=-1)

    write_log(args.log_file, f"[{timestamp()}] Training start")
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    best_macro_auc = 0.0
    patience_count = 0

    # 训练轮次
    for epoch in range(args.epoch):
        model.train()
        train_loss = 0.
        data_count = 0
        with tqdm(train_loader, unit="it") as train_bar:
            for data in train_bar:
                optimizer.zero_grad()
                forward_dict = forward_model(model, data, device)
                output = forward_dict["output"]
                score = forward_dict["score"]
                seq_mask = forward_dict["seq_mask"]
                output = output[seq_mask].flatten()
                score_true = score[:, 1:][seq_mask].float().flatten()
                train_len = len(score_true)
                # 计算不同模型对应的损失
                if model.name == "GraDKT":
                    loss = kt_loss(output, score_true)
                    if forward_dict["r_target"] is not None:
                        r_output = forward_dict["r_target"][seq_mask].flatten().unsqueeze(-1)
                        r_true = forward_dict["g_target"][seq_mask].float().flatten().unsqueeze(-1)
                        loss += args.alpha * r_loss(r_output, r_true)
                    if forward_dict["inter_label"] is not None:
                        loss += args.beta * cl_loss(forward_dict["inter_cossim"], forward_dict["inter_label"])
                    if "graph_cl_loss" in forward_dict:
                        loss += args.gamma * forward_dict["graph_cl_loss"]
                        #  新增重建损失项（score 重建）
                    if "score_recon" in forward_dict:
                        score_recon_pred = forward_dict["score_recon"][:, 1:][seq_mask].flatten()
                        recon_loss = F.binary_cross_entropy(score_recon_pred, score_true)
                        loss += args.recon_weight * recon_loss  # ←← 加权整合入总损失                  
                loss.backward()
                optimizer.step()
                data_count += train_len
                train_loss += loss.item() * train_len
                train_bar.set_postfix(loss=f"{train_loss / data_count:.6f}")

        # 在验证集上评估
        result_dict, result_str = predict(model, valid_loader, valid_set.correct_rate, device)
        val_loss = result_dict["loss"]
        macro_auc = result_dict.get("Macro-AUC", 0.0)
        log_str = f"[{timestamp()}] epoch={epoch + 1:2d} | {result_str}"
        write_log(args.log_file, log_str)

        # 更新最佳 Macro-AUC（注意：AUC 越高越好）
        if macro_auc > best_macro_auc:
            patience_count = 0
            best_model_file = os.path.join(args.save_path, "best.pt")
            torch.save(model.state_dict(), best_model_file, pickle_protocol=4)
            best_macro_auc = macro_auc
        else:
            patience_count += 1
            if patience_count >= max_patience:
                write_log(args.log_file, f"Patience count reached at {max_patience}. Early stopped.")
                break
            else:
                write_log(args.log_file, f"Patience count: {patience_count}/{max_patience}. Best Macro-AUC so far: {best_macro_auc:.4f}, current: {macro_auc:.4f}")

    return best_macro_auc


def test(args):
    device = args.device
    test_set = KTDataset(args.dataset_path, ["test.json"], args.max_concept_len)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, collate_fn=pad_collate)

    model = load_model(args.model, args)
    model = model.to(device)
    model_path = os.path.join(args.save_path, "best.pt")
    model.load_state_dict(torch.load(model_path))

    result_dict, result_str = predict(model, test_loader, test_set.correct_rate, device)
    log_str = f"[{timestamp()}] model={args.model} exp={args.exp_name}\n{result_str}\n"
    write_log(args.log_file, log_str)
    write_log(args.result_file, log_str.replace("\n", " "), console=False)
    return result_dict


def forward_model(model, data, device):
    (user, question, concept, score, option, answer, unchosen,
     pos_score, pos_option, neg_score, neg_option) = data
    user, question, concept, score = user.to(device), question.to(device), concept.to(device), score.to(device)
    option, answer, unchosen = option.to(device), answer.to(device), unchosen.to(device)
    pos_score, pos_option, neg_score, neg_option = pos_score.to(device), pos_option.to(device), neg_score.to(device), neg_option.to(device)

    seq_mask = torch.ne(score, -1)[:, 1:]
    forward_dict = {"score": score, "question": question, "seq_mask": seq_mask}
    if model.name == "GraDKT":
        output, r_target, g_target, inter_cossim, inter_label, graph_cl_loss, score_recon_pred = model(question, concept, score, option, unchosen,
                                                                                      pos_score, pos_option, neg_score, neg_option)
        forward_dict["r_target"] = r_target
        forward_dict["g_target"] = g_target
        forward_dict["inter_cossim"] = inter_cossim
        forward_dict["inter_label"] = inter_label
        forward_dict["graph_cl_loss"] = graph_cl_loss
        forward_dict["score_recon"] = score_recon_pred
    else:
        print(f"forward() for model {model.name} not found.")
        exit()
    forward_dict["output"] = output
    return forward_dict


def predict(model, pred_loader, correct_rate, device):
    model.eval()
    pred_loss = 0.
    data_count = 0
    kt_loss = BCELoss()
    ot_loss = CrossEntropyLoss(ignore_index=-1)
    pos_weight = torch.tensor((args.concept - args.top_k) / args.top_k, device=device)
    r_loss = BCEWithLogitsLoss(pos_weight=pos_weight)

    y_true, y_hat, y_pred, y_question = [], [], [], []
    with tqdm(pred_loader, unit="it") as pred_bar, torch.no_grad():
        for data in pred_bar:
            forward_dict = forward_model(model, data, device)
            output = forward_dict["output"]
            score = forward_dict["score"]
            question = forward_dict["question"]
            seq_mask = forward_dict["seq_mask"]
            output = output[seq_mask].flatten()
            score_true = score[:, 1:][seq_mask].float().flatten()
            question_valid = question[:, 1:][seq_mask].float().flatten()
            pred_len = len(score_true)
            output_pred = (output >= 0.5).int()
            y_true += score_true.cpu().tolist()
            y_hat += output.cpu().tolist()
            y_pred += output_pred.cpu().tolist()
            y_question += question_valid.cpu().tolist()
            if model.name == "GraDKT":
                if forward_dict["r_target"] is not None:
                    r_output = forward_dict["r_target"][seq_mask].flatten().unsqueeze(-1)
                    r_true = forward_dict["g_target"][seq_mask].float().flatten().unsqueeze(-1)
                    loss = kt_loss(output, score_true) + args.alpha * r_loss(r_output, r_true)
                else:
                    loss = kt_loss(output, score_true)

                if "score_recon" in forward_dict:
                    score_recon_pred = forward_dict["score_recon"][:, 1:][seq_mask].flatten()
                    recon_loss = F.binary_cross_entropy(score_recon_pred, score_true)
                    loss += args.recon_weight * recon_loss
            elif model.name == "DTransformer":
                loss = forward_dict["loss"]
            elif model.name == "DP_DKT":
                ot_output = forward_dict["ot_output"][seq_mask].reshape(-1, args.option)
                option_true = forward_dict["option"][:, 1:][seq_mask].flatten()
                loss = args.lamb * kt_loss(output, score_true) + (1 - args.lamb) * ot_loss(ot_output, option_true)
            else:
                loss = kt_loss(output, score_true)
            data_count += pred_len
            pred_loss += loss.item() * pred_len
            pred_bar.set_postfix(loss=f"{pred_loss / data_count:.6f}")

    y_majority = (correct_rate[y_question] >= 0.5).int().tolist()
    y_minority = (correct_rate[y_question] < 0.5).int().tolist()
    result_dict, result_str = calc_metric(y_true, y_hat, y_pred, y_majority, y_minority, pred_loss, data_count)
    return result_dict, result_str


if __name__ == "__main__":
    args = MainArgParser().parse_args()
    args.lr = float(args.lr)
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model_list = ["GraDKT"]
    if args.model not in model_list:
        raise Exception(f"Model Not Found: {args.model} is not in {model_list}.")

    set_seed(args.seed)
    set_config(args)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    write_log(args.log_file, args)

    start_total = time()
    best_macro_auc = train(args)
    write_log(args.log_file, f"[{timestamp()}] Best Macro-AUC on validation: {best_macro_auc:.4f}")
    
    test_result_dict = test(args)

    total_time = int(time() - start_total)
    time_str = f"[{timestamp()}] All done. ({total_time // 3600:02d}:{(total_time % 3600) // 60:02d}:{total_time % 60:02d})"
    write_log(args.log_file, time_str)

    