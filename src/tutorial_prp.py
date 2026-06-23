import json
import os
import sys
import time
import copy
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Union, Optional
import warnings

import torch
import numpy as np
import matplotlib
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from torch_geometric.loader import DataLoader
from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction

sys.path.insert(0, "../")

import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.loss import (
    masked_mse_loss,
    criterion_neg_log_bernoulli,
    masked_relative_error,
)
from scgpt.tokenizer import tokenize_batch, pad_batch, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import load_pretrained, set_seed, map_raw_id_to_vocab_id, compute_perturbation_metrics

# nano-scgpt
from scGPT_tokenizer import scGPTTokenizer
from perturbation_data import PerturbationDataSplitter, PerturbationDataset
from model import scGPTForPerturbationResponsePrediction

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")

def load_og_model(load_model: str, pert_data, special_tokens: List[str], device: torch.device):
    model_dir = Path(load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    genes = pert_data.adata.var["gene_name"].tolist()
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )
    n_genes = len(genes)
    ntokens = len(vocab)  # size of vocabulary

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
    vocab.set_default_index(vocab["<pad>"])
    model = TransformerGenerator(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        pert_pad_id=pert_pad_id,
        use_fast_transformer=use_fast_transformer,
    )
    load_pretrained(model, torch.load(model_file, map_location=device), verbose=False)    
    model.to(device)

    return model, vocab, gene_ids, n_genes


def load_nano_model(pert_data, seed, batch_size):
    tokenizer = scGPTTokenizer.from_pretrained("scGPT_human")
    tokenizer.max_length = 1536
    adata = pert_data.adata
    adata.var['gene_symbol'] = adata.var['gene_name']
    data_splitter = PerturbationDataSplitter(adata, tokenizer, seed=seed)
    train_adata, val_adata, test_adata = data_splitter.get_train_val_test()
    train_dataset = PerturbationDataset(train_adata, tokenizer, split='train')
    test_dataset = PerturbationDataset(test_adata, tokenizer, split='test')
    val_dataset = PerturbationDataset(val_adata, tokenizer, split='val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)


    model = scGPTForPerturbationResponsePrediction.from_pretrained("scGPT_human")
    model.gene_ids = train_dataset.gene_ids

    return model, train_loader, val_loader, test_loader


def eval_perturb(
    loader: DataLoader, model: TransformerGenerator, nano_model: nn.Module, device: torch.device
) -> Dict:
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    nano_model.eval()
    nano_model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []

    for itr, batch in enumerate(loader):
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            p = model.pred_perturb(
                batch,
                include_zero_gene=include_zero_gene,
                gene_ids=gene_ids,
            )
            print(f"pred shape: {p.shape}, truth shape: {batch.y.shape}")
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())

            nano_p = nano_model.pred_perturb(
                batch,
                include_zero_gene=include_zero_gene,
                gene_ids=gene_ids,
            )
            print(f"nano pred shape: {nano_p.shape}, truth shape: {batch.y.shape}")
            
            import pdb; pdb.set_trace();

            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])
            
            break;

    # all genes
    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred) # (N, n_genes)
    truth = torch.stack(truth) # (N, n_genes)
    results["pred"] = pred.detach().cpu().numpy()
    results["truth"] = truth.detach().cpu().numpy()

    pred_de = torch.stack(pred_de) # (N, n_de_genes)
    truth_de = torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy()
    results["truth_de"] = truth_de.detach().cpu().numpy()

    return results


def train(model: nn.Module, train_loader: torch.utils.data.DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    criterion = masked_mse_loss
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    model.train()
    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, n_genes)
        target_gene_values = batch_data.y  # (batch_size, n_genes)

        if include_zero_gene in ["all", "batch-wise"]:
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            # sample input_gene_id
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    :max_seq_len
                ]
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

        with torch.cuda.amp.autocast(enabled=amp):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
            )
            output_values = output_dict["mlm_output"]

            nano_output_values = nano_model(
                mapped_input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                pert_labels=input_pert_flags,
            )

            import pdb; pdb.set_trace();

            masked_positions = torch.ones_like(
                input_values, dtype=torch.bool
            )  # Use all
            loss = loss_mse = criterion(output_values, target_values, masked_positions)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        # torch.cuda.empty_cache()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
            )
            total_loss = 0
            total_mse = 0
            start_time = time.time()


    best_val_loss = float("inf")
    best_val_corr = 0
    best_model = None
    patience = 0

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loader = pert_data.dataloader["train_loader"]
        valid_loader = pert_data.dataloader["val_loader"]

        if epoch == 1:
            val_res = eval_perturb(valid_loader, model, device)
            val_metrics = compute_perturbation_metrics(
                val_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
            )
            logger.info(f"val_metrics before training at epoch {epoch}: ")
            logger.info(val_metrics)

        print(f"Epoch {epoch} training...")
        train(
            model,
            train_loader,
        )

        print(f"Epoch {epoch} evaluating...")
        val_res = eval_perturb(valid_loader, model, device)
        val_metrics = compute_perturbation_metrics(
            val_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
        )
        logger.info(f"val_metrics at epoch {epoch}: ")
        logger.info(val_metrics)

        elapsed = time.time() - epoch_start_time
        logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ")

        val_score = val_metrics["pearson"]
        if val_score > best_val_corr:
            best_val_corr = val_score
            best_model = copy.deepcopy(model)
            logger.info(f"Best model with score {val_score:5.4f}")
            patience = 0
        else:
            patience += 1
            if patience >= early_stop:
                logger.info(f"Early stop at epoch {epoch}")
                break

        # torch.save(
        #     model.state_dict(),
        #     save_dir / f"model_{epoch}.pt",
        # )

        scheduler.step()



    # %% [code] cell 11
    torch.save(best_model.state_dict(), save_dir / "best_model.pt")


if __name__ == "__main__":
    seed = 42
    set_seed(seed)
    
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    pad_value = 0  # for padding values
    pert_pad_id = 0
    include_zero_gene = "all"
    max_seq_len = 1536

    # settings for training
    MLM = True  # whether to use masked language modeling, currently it is always on.
    CLS = False  # celltype classification objective
    CCE = False  # Contrastive cell embedding objective
    MVC = False  # Masked value prediction for cell embedding
    ECS = False  # Elastic cell similarity objective
    amp = True
    load_model = "../pretrained_weights/scGPT_human"
    load_param_prefixs = [
        "encoder",
        "value_encoder",
        "transformer_encoder",
    ]

    # settings for optimizer
    lr = 1e-4  # or 1e-4
    batch_size = 64
    eval_batch_size = 64
    # batch_size = 4
    # eval_batch_size = 4
    epochs = 15
    schedule_interval = 1
    early_stop = 10

    # settings for the model
    embsize = 512  # embedding dimension
    d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 8  # number of heads in nn.MultiheadAttention
    n_layers_cls = 3
    dropout = 0  # dropout probability
    use_fast_transformer = True  # whether to use fast transformer

    # logging
    log_interval = 100

    # dataset and evaluation choices
    data_name = "adamson"
    split = "simulation"
    if data_name == "norman":
        perts_to_plot = ["SAMD1+ZBTB1"]
    elif data_name == "adamson":
        perts_to_plot = ["KCTD16+ctrl"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"======Using device: {device}======")

    save_dir = Path(f"./save/dev_perturb_{data_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving to {save_dir}")
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("Loading PertData...")
    pert_data = PertData("./data")
    pert_data.load(data_name=data_name)
    pert_data.prepare_split(split=split, seed=1)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)

    print("Loading og model ...")
    model, vocab, gene_ids, n_genes = load_og_model(load_model, pert_data, special_tokens, device)

    print("Loading nano-scgpt model and tokenizer...")
    nano_model, nano_train_loader, nano_val_loader, nano_test_loader = load_nano_model(pert_data, seed, batch_size)

    eval_perturb(pert_data.dataloader["test_loader"], model, nano_model, device)