"""
train_perturbation.py
OG scGPT perturbation fine-tuning — refactored from Tutorial_Perturbation.ipynb
for debugging. Structurally faithful to the original; zero behaviour changes.

Usage:
    python train_perturbation.py --data_name adamson --load_model ../save/scGPT_human
    python train_perturbation.py --data_name norman  --load_model ../save/scGPT_human
"""

import argparse
import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import matplotlib
import numpy as np
import torch
from torch import device, nn
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torchtext._torchtext import Vocab as VocabPybind
from torchtext.vocab import Vocab

from gears import GEARS, PertData
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction

sys.path.insert(0, "../")

import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.loss import (
    criterion_neg_log_bernoulli,
    masked_mse_loss,
    masked_relative_error,
)
from scgpt.tokenizer import pad_batch, tokenize_and_pad_batch, tokenize_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import (
    compute_perturbation_metrics,
    map_raw_id_to_vocab_id,
    set_seed,
    load_pretrained
)

from nano_scgpt.scGPT_tokenizer import scGPTTokenizer
from nano_scgpt.perturbation_data import PerturbationDataSplitter, PerturbationDataset

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="scGPT perturbation fine-tuning")

    # data
    p.add_argument("--data_name", default="adamson", choices=["adamson", "norman"])
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--split", default="simulation")

    # model loading
    p.add_argument("--load_model", default="../pretrained_weights/scGPT_human",
                   help="Path to pretrained model dir. Set to '' to train from scratch.")
    p.add_argument("--load_param_prefixs", nargs="+",
                   default=["encoder", "value_encoder", "transformer_encoder"])

    # data processing
    p.add_argument("--pad_token", default="<pad>")
    p.add_argument("--pad_value", type=int, default=0)
    p.add_argument("--pert_pad_id", type=int, default=0)
    p.add_argument("--include_zero_gene", default="all",
                   choices=["all", "batch-wise"])
    p.add_argument("--max_seq_len", type=int, default=1536)

    # optimiser
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--schedule_interval", type=int, default=1)
    p.add_argument("--early_stop", type=int, default=10)

    # model arch (overridden by checkpoint if load_model is set)
    p.add_argument("--embsize", type=int, default=512)
    p.add_argument("--d_hid", type=int, default=512)
    p.add_argument("--nlayers", type=int, default=12)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--n_layers_cls", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--use_fast_transformer", action="store_true", default=True)

    # objectives (kept for parity; all off except MLM)
    p.add_argument("--MLM", action="store_true", default=True)
    p.add_argument("--CLS", action="store_true", default=False)
    p.add_argument("--CCE", action="store_true", default=False)
    p.add_argument("--MVC", action="store_true", default=False)
    p.add_argument("--ECS", action="store_true", default=False)
    p.add_argument("--amp", action="store_true", default=True)

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--save_dir", default=None,
                   help="Override save directory. Default: ./save/dev_perturb_<name>-<time>/")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(args, prepare_split=True) -> PertData:
    pert_data = PertData(args.data_dir)
    pert_data.load(data_name=args.data_name)
    if prepare_split:
        pert_data.prepare_split(split=args.split, seed=1)
        pert_data.get_dataloader(
            batch_size=args.batch_size,
            test_batch_size=args.eval_batch_size,
        )
    return pert_data


def load_nano_data(adata, batch_size, seed):
    adata.var['gene_symbol'] = adata.var['gene_name']

    tokenizer = scGPTTokenizer.from_pretrained("scGPT_human")
    tokenizer.max_length = 1536
    data_splitter = PerturbationDataSplitter(adata, tokenizer, seed=seed)
    train_adata, val_adata, test_adata = data_splitter.get_train_val_test()

    train_dataset = PerturbationDataset(train_adata, tokenizer, split='train')
    test_dataset = PerturbationDataset(test_adata, tokenizer, split='test')
    val_dataset = PerturbationDataset(val_adata, tokenizer, split='val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)

    return train_loader, val_loader, test_loader, tokenizer

def build_vocab(args, pert_data: PertData, logger):
    special_tokens = [args.pad_token, "<cls>", "<eoc>"]

    if args.load_model:
        model_dir = Path(args.load_model)
        vocab_file = model_dir / "vocab.json"
        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        pert_data.adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1
            for gene in pert_data.adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
    else:
        genes = pert_data.adata.var["gene_name"].tolist()
        vocab = Vocab(VocabPybind(genes + special_tokens, None))

    vocab.set_default_index(vocab[args.pad_token])

    genes = pert_data.adata.var["gene_name"].tolist()
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab[args.pad_token] for gene in genes],
        dtype=int,
    )
    return vocab, genes, gene_ids


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(args, vocab, logger) -> TransformerGenerator:
    # Override arch params from checkpoint if available
    if args.load_model:
        model_dir = Path(args.load_model)
        with open(model_dir / "args.json") as f:
            model_configs = json.load(f)
        logger.info(
            f"Resume model from {model_dir / 'best_model.pt'}, "
            f"model args will override config."
        )
        embsize      = model_configs["embsize"]
        nhead        = model_configs["nheads"]
        d_hid        = model_configs["d_hid"]
        nlayers      = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]
    else:
        embsize      = args.embsize
        nhead        = args.nhead
        d_hid        = args.d_hid
        nlayers      = args.nlayers
        n_layers_cls = args.n_layers_cls

    model = TransformerGenerator(
        ntoken=len(vocab),
        d_model=embsize,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=args.dropout,
        pad_token=args.pad_token,
        pad_value=args.pad_value,
        pert_pad_id=args.pert_pad_id,
        use_fast_transformer=args.use_fast_transformer,
    )

    if args.load_model:
        model_file = Path(args.load_model) / "best_model.pt"
        load_pretrained(model, torch.load(model_file, map_location='cpu'), verbose=False)
    else:
        logger.info("No pretrained weights found. Training from scratch.")
    
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion,
    optimizer,
    scaler,
    args,
    epoch: int,
    n_genes: int,
    gene_ids: np.ndarray,
    device: torch.device,
    logger,
) -> None:
    model.train()
    total_loss = 0.0
    total_mse  = 0.0
    start_time = time.time()
    num_batches = len(train_loader)

    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)

        x: torch.Tensor = batch_data.x          # (batch_size * n_genes, 2)
        ori_gene_values  = x[:, 0].view(batch_size, n_genes)
        pert_flags       = x[:, 1].long().view(batch_size, n_genes)
        target_gene_values = batch_data.y        # (batch_size, n_genes)

        # ---- gene subsampling (the 70%-dropout step) ----------------------
        if args.include_zero_gene in ["all", "batch-wise"]:
            if args.include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )

            if len(input_gene_ids) > args.max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    : args.max_seq_len
                ]

            input_values       = ori_gene_values[:, input_gene_ids]
            input_pert_flags   = pert_flags[:, input_gene_ids]
            target_values      = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )
        # -------------------------------------------------------------------

        with torch.cuda.amp.autocast(enabled=args.amp):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=args.CLS,
                CCE=args.CCE,
                MVC=args.MVC,
                ECS=args.ECS,
            )
            output_values = output_dict["mlm_output"]
            masked_positions = torch.ones_like(input_values, dtype=torch.bool)
            loss = loss_mse = criterion(output_values, target_values, masked_positions)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if w:
                logger.warning(
                    f"Infinite gradient detected. Scale={scaler.get_scale()}. "
                    "Safe to ignore if not recurring."
                )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mse  += loss_mse.item()

        if batch % args.log_interval == 0 and batch > 0:
            lr_now = optimizer.param_groups[0]["lr"]
            ms_per_batch = (time.time() - start_time) * 1000 / args.log_interval
            cur_loss = total_loss / args.log_interval
            cur_mse  = total_mse  / args.log_interval
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr_now:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
            )
            total_loss = 0.0
            total_mse  = 0.0
            start_time = time.time()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_perturb(
    loader: DataLoader,
    model: TransformerGenerator,
    device: torch.device,
    include_zero_gene: str,
    gene_ids: np.ndarray,
    amp: bool,
) -> Dict:
    model.eval()
    model.to(device)

    pert_cat, pred, truth, pred_de, truth_de = [], [], [], [], []

    for itr, batch in enumerate(loader):
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            p = model.pred_perturb(
                batch,
                include_zero_gene=include_zero_gene,
                gene_ids=gene_ids,
            )
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())

            for i, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[i, de_idx])
                truth_de.append(t[i, de_idx])

    results = {}
    results["pert_cat"] = np.array(pert_cat)
    pred  = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"]  = pred.detach().cpu().numpy().astype(float)
    results["truth"] = truth.detach().cpu().numpy().astype(float)

    pred_de  = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"]  = pred_de.detach().cpu().numpy().astype(float)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(float)

    return results


# ---------------------------------------------------------------------------
# Prediction helpers (for plotting)
# ---------------------------------------------------------------------------

def predict(
    model: TransformerGenerator,
    pert_list: List[List[str]],
    pert_data: PertData,
    gene_ids: np.ndarray,
    include_zero_gene: str,
    eval_batch_size: int,
    amp: bool,
    pool_size: Optional[int] = None,
) -> Dict:
    adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    if pool_size is None:
        pool_size = len(ctrl_adata.obs)
    gene_list = pert_data.gene_names.values.tolist()

    for pert in pert_list:
        for g in pert:
            if g not in gene_list:
                raise ValueError(
                    f"Gene '{g}' not in perturbation graph. "
                    "Choose from pert_data.gene_names."
                )

    device = next(model.parameters()).device
    model.eval()
    results_pred = {}

    with torch.no_grad():
        for pert in pert_list:
            cell_graphs = create_cell_graph_dataset_for_prediction(
                pert, ctrl_adata, gene_list, device, num_samples=pool_size
            )
            loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)
            preds = []
            for batch_data in loader:
                pred_gene_values = model.pred_perturb(
                    batch_data,
                    include_zero_gene,
                    gene_ids=gene_ids,
                    amp=amp,
                )
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            results_pred["_".join(pert)] = np.mean(
                preds.detach().cpu().numpy(), axis=0
            )

    return results_pred


def plot_perturbation(
    model: nn.Module,
    query: str,
    pert_data: PertData,
    gene_ids: np.ndarray,
    include_zero_gene: str,
    eval_batch_size: int,
    amp: bool,
    save_file: Optional[str] = None,
    pool_size: Optional[int] = None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

    adata = pert_data.adata
    gene2idx = pert_data.node_map
    cond2name = dict(adata.obs[["condition", "condition_name"]].values)
    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

    de_idx = [
        gene2idx[gene_raw2id[i]]
        for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]
    genes = [
        gene_raw2id[i]
        for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]
    truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]

    parts = query.split("+")
    if parts[1] == "ctrl":
        pred_dict = predict(
            model, [[parts[0]]], pert_data, gene_ids,
            include_zero_gene, eval_batch_size, amp, pool_size=pool_size,
        )
        pred = pred_dict[parts[0]][de_idx]
    else:
        pred_dict = predict(
            model, [parts], pert_data, gene_ids,
            include_zero_gene, eval_batch_size, amp, pool_size=pool_size,
        )
        pred = pred_dict["_".join(parts)][de_idx]

    ctrl_means = (
        adata[adata.obs["condition"] == "ctrl"].to_df().mean()[de_idx].values
    )
    pred  = pred  - ctrl_means
    truth = truth - ctrl_means

    fig, ax = plt.subplots(figsize=[16.5, 4.5])
    plt.title(query)
    plt.boxplot(truth, showfliers=False, medianprops=dict(linewidth=0))
    for i in range(pred.shape[0]):
        plt.scatter(i + 1, pred[i], color="red")
    plt.axhline(0, linestyle="dashed", color="green")
    ax.xaxis.set_ticklabels(genes, rotation=90)
    plt.ylabel("Change in Gene Expression over Control", labelpad=10)
    plt.tick_params(axis="x", which="major", pad=5)
    plt.tick_params(axis="y", which="major", pad=5)
    sns.despine()

    if save_file:
        fig.savefig(save_file, bbox_inches="tight", transparent=False)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save dir
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path(
            f"./save/dev_perturb_{args.data_name}-{time.strftime('%b%d-%H-%M')}/"
        )
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {save_dir}")

    # Logger
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Args: {vars(args)}")

    # ---- Data ---------------------------------------------------------------
    prepare_split_in_data_loading = False
    logger.info("Loading perturbation data...")
    pert_data = load_data(args, prepare_split=prepare_split_in_data_loading)

    nano_train_loader, nano_val_loader, nano_test_loader, tokenizer = load_nano_data(
        pert_data.adata, args.batch_size, seed=args.seed
    )

    # Manually align splits
    if not prepare_split_in_data_loading:
        test_perts = sorted([p for p in nano_test_loader.dataset.perturbations if p != "ctrl"])
        pert_data.prepare_split(only_test_set_perts=True, test_pert_genes=test_perts, seed=args.seed)
        pert_data.get_dataloader(
            batch_size=args.batch_size,
            test_batch_size=args.eval_batch_size,
        )

    # yours
    print('Checking test perturbations...')
    print(sorted(nano_test_loader.dataset.perturbations))
    # OG
    print(sorted(pert_data.set2conditions['test']))

    vocab, genes, gene_ids = build_vocab(args, pert_data, logger)
    n_genes = len(genes)
    logger.info(f"n_genes={n_genes}, vocab_size={len(vocab)}")

    # ---- Model --------------------------------------------------------------
    logger.info("Building model...")
    model = build_model(args, vocab, logger)
    model.to(device)

    # ---- Optimiser ----------------------------------------------------------
    criterion     = masked_mse_loss
    criterion_cls = nn.CrossEntropyLoss()
    optimizer     = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler     = torch.optim.lr_scheduler.StepLR(
        optimizer, args.schedule_interval, gamma=0.9
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # ---- Perturbation plot target -------------------------------------------
    if args.data_name == "norman":
        perts_to_plot = ["SAMD1+ZBTB1"]
    else:
        perts_to_plot = ["KCTD16+ctrl"]

    # ---- Training loop ------------------------------------------------------
    best_val_loss = float("inf")
    best_val_corr = 0.0
    best_model    = None
    patience      = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loader = pert_data.dataloader["train_loader"]
        valid_loader = pert_data.dataloader["val_loader"]

        train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            args, epoch, n_genes, gene_ids, device, logger,
        )

        val_res = eval_perturb(
            valid_loader, model, device,
            args.include_zero_gene, gene_ids, args.amp,
        )
        val_metrics = compute_perturbation_metrics(
            val_res,
            pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"],
        )
        logger.info(f"val_metrics at epoch {epoch}: {val_metrics}")
        logger.info(f"| end of epoch {epoch:3d} | time: {time.time()-epoch_start:5.2f}s |")

        val_score = val_metrics["pearson"]
        if val_score > best_val_corr:
            best_val_corr = val_score
            best_model    = copy.deepcopy(model)
            logger.info(f"Best model — pearson={val_score:.4f}")
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        scheduler.step()

    # ---- Save best model ----------------------------------------------------
    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    logger.info(f"Saved best model to {save_dir / 'best_model.pt'}")

    # ---- Perturbation plots -------------------------------------------------
    for p in perts_to_plot:
        plot_perturbation(
            best_model, p, pert_data, gene_ids,
            args.include_zero_gene, args.eval_batch_size, args.amp,
            save_file=str(save_dir / f"{p}.png"),
            pool_size=300,
        )

    # ---- Test evaluation ----------------------------------------------------
    test_loader = pert_data.dataloader["test_loader"]
    test_res    = eval_perturb(
        test_loader, best_model, device,
        args.include_zero_gene, gene_ids, args.amp,
    )
    test_metrics = compute_perturbation_metrics(
        test_res,
        pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"],
    )
    logger.info(f"test_metrics: {test_metrics}")
    print("test_metrics:", test_metrics)

    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # ---- Deeper / subgroup analysis -----------------------------------------
    deeper_res      = deeper_analysis(pert_data.adata, test_res)
    non_dropout_res = non_dropout_analysis(pert_data.adata, test_res)

    metrics              = ["pearson_delta", "pearson_delta_de"]
    metrics_non_dropout  = [
        "pearson_delta_top20_de_non_dropout",
        "pearson_top20_de_non_dropout",
    ]
    subgroup_analysis = {}
    for name in pert_data.subgroup["test_subgroup"].keys():
        subgroup_analysis[name] = {m: [] for m in metrics + metrics_non_dropout}

    for name, pert_list in pert_data.subgroup["test_subgroup"].items():
        for pert in pert_list:
            for m in metrics:
                subgroup_analysis[name][m].append(deeper_res[pert][m])
            for m in metrics_non_dropout:
                subgroup_analysis[name][m].append(non_dropout_res[pert][m])

    for name, result in subgroup_analysis.items():
        for m, vals in result.items():
            logger.info(f"test_{name}_{m}: {np.mean(vals):.4f}")

    with open(save_dir / "subgroup_analysis.json", "w") as f:
        json.dump(
            {n: {m: float(np.mean(v)) for m, v in r.items()}
             for n, r in subgroup_analysis.items()},
            f, indent=2,
        )
    logger.info("Done.")


if __name__ == "__main__":
    main()
