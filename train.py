import copy
import os
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import Dataset
from model.birnn import NMTModel
from model.embeddings import Embeddings
from model.generator import Generator
from model.multi_head_attention import MultiHeadedAttention
from model.positionwise_feed_forward import PositionwiseFeedForward
from model.postional_encoding import PositionalEncoding
from model.transformer import Decoder, DecoderLayer, Encoder, EncoderLayer, Transformer
from utils.bookkeeping import make_train_state, update_train_state
from utils.helpers import handle_dirs, set_seed_everywhere
from utils.logger import Logger


def normalize_sizes(y_pred, y_true):
    """Normalize tensor sizes

    Args:
        y_pred (torch.Tensor): the output of the model
            If a 3-dimensional tensor, reshapes to a matrix
        y_true (torch.Tensor): the target predictions
            If a matrix, reshapes to be a vector
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true


def compute_accuracy(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    _, y_pred_indices = y_pred.max(dim=1)

    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()

    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100


def sequence_loss(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Construct a model from hyperparameters."

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout, 512)

    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
    """A generator function which wraps the PyTorch DataLoader."""
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )

    for data_dict in dataloader:
        lengths = data_dict["x_source_length"].numpy()
        sorted_length_indices = lengths.argsort()[::-1].tolist()

        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)
        yield out_data_dict


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


if __name__ == "__main__":
    args = Namespace(
        dataset_csv="data/lyrics.csv",
        vectorizer_file="vectorizer.json",
        model_state_file="model.pth",
        save_dir="model.storage/cyrillc_to_mongolian",
        reload_from_files=True,
        expand_filepaths_to_save_dir=True,
        cuda=False,
        seed=1337,
        learning_rate=5e-4,
        batch_size=16,
        num_epochs=10,
        early_stopping_criteria=5,
        source_embedding_size=16,
        target_embedding_size=16,
        encoding_size=32,
        catch_keyboard_interrupt=True,
    )

    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)
        args.model_state_file = os.path.join(args.save_dir, args.model_state_file)
        Logger.logi(
            __name__,
            "Expanded filepaths: \n"
            f"\t{args.vectorizer_file}\n"
            f"\t{args.model_state_file}",
        )

    if not torch.cuda.is_available():
        args.cuda = False
        Logger.logi(__name__, "CUDA is not available")

    args.device = torch.device("cuda" if args.cuda else "cpu")
    Logger.logi(__name__, f"Using CUDA: {args.cuda}")

    set_seed_everywhere(args.seed, args.cuda)
    handle_dirs(args.save_dir)

    if args.reload_from_files and os.path.exists(args.vectorizer_file):
        dataset = Dataset.load_dataset_and_load_vectorizer(
            args.dataset_csv, args.vectorizer_file
        )
    else:
        dataset = Dataset.load_dataset_and_make_vectorizer(args.dataset_csv)
        dataset.save_vectorizer(args.vectorizer_file)

    vectorizer = dataset.get_vectorizer()

    model = NMTModel(
        source_vocab_size=len(vectorizer.source_vocab),
        source_embedding_size=args.source_embedding_size,
        target_vocab_size=len(vectorizer.target_vocab),
        target_embedding_size=args.target_embedding_size,
        encoding_size=args.encoding_size,
        target_bos_index=vectorizer.target_vocab.begin_seq_index,
    )

    if args.reload_from_files and os.path.exists(args.model_state_file):
        model.load_state_dict(torch.load(args.model_state_file))
        Logger.logi(__name__, f"Reload model state from {args.model_state_file}")
    else:
        Logger.logi(__name__, "New model")

    model.to(args.device)

    mask_index = vectorizer.target_vocab.mask_index
    train_state = make_train_state(args)

    epoch_bar = tqdm(desc="training routine", total=args.num_epochs, position=0)

    dataset.set_split("train")
    train_bar = tqdm(
        desc="split = train",
        total=dataset.get_num_batches(args.batch_size),
        position=1,
        leave=True,
        bar_format='{l_bar}{r_bar}'
    )
    dataset.set_split("val")
    val_bar = tqdm(
        desc="split = val",
        total=dataset.get_num_batches(args.batch_size),
        position=1,
        leave=True,
        bar_format='{l_bar}{r_bar}'
    )

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.5, patience=1
    )

    try:
        for epoch_index in range(args.num_epochs):
            train_state["epoch_index"] = epoch_index

            # Iterate over training dataset

            # setup: batch generator, set loss and acc to 0, set train mode on
            dataset.set_split("train")
            batch_generator = generate_batches(
                dataset, batch_size=args.batch_size, device=args.device
            )
            running_loss = 0.0
            running_acc = 0.0
            model.train()

            for batch_index, batch_dict in enumerate(batch_generator):
                # the training routine is these 5 steps:

                # --------------------------------------
                # step 1. zero the gradients
                optimizer.zero_grad()

                # step 2. compute the output
                y_pred = model(
                    batch_dict["x_source"],
                    batch_dict["x_source_length"],
                    batch_dict["x_target"],
                )

                # step 3. compute the loss
                loss = sequence_loss(y_pred, batch_dict["y_target"], mask_index)

                # step 4. use loss to produce gradients
                loss.backward()

                # step 5. use optimizer to take gradient step
                optimizer.step()
                # -----------------------------------------
                # compute the running loss and running accuracy
                running_loss += (loss.item() - running_loss) / (batch_index + 1)

                acc_t = compute_accuracy(y_pred, batch_dict["y_target"], mask_index)
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                # update bar
                train_bar.set_postfix(
                    loss=running_loss, acc=running_acc, epoch=epoch_index
                )
                train_bar.update()

            train_state["train_loss"].append(running_loss)
            train_state["train_acc"].append(running_acc)

            # Iterate over val dataset

            # setup: batch generator, set loss and acc to 0; set eval mode on
            dataset.set_split("val")
            batch_generator = generate_batches(
                dataset, batch_size=args.batch_size, device=args.device
            )
            running_loss = 0.0
            running_acc = 0.0
            model.eval()

            for batch_index, batch_dict in enumerate(batch_generator):
                # compute the output
                y_pred = model(
                    batch_dict["x_source"],
                    batch_dict["x_source_length"],
                    batch_dict["x_target"],
                )

                # step 3. compute the loss
                loss = sequence_loss(y_pred, batch_dict["y_target"], mask_index)

                # compute the running loss and accuracy
                running_loss += (loss.item() - running_loss) / (batch_index + 1)

                acc_t = compute_accuracy(y_pred, batch_dict["y_target"], mask_index)
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                # Update bar
                val_bar.set_postfix(
                    loss=running_loss, acc=running_acc, epoch=epoch_index
                )
                val_bar.update()

            train_state["val_loss"].append(running_loss)
            train_state["val_acc"].append(running_acc)

            train_state = update_train_state(
                args=args, model=model, train_state=train_state
            )

            scheduler.step(train_state["val_loss"][-1])

            if train_state["stop_early"]:
                Logger.logi(__name__, "Early stopping.")
                break

            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.set_postfix(best_val=train_state["early_stopping_best_val"])
            epoch_bar.update()

    except KeyboardInterrupt:
        Logger.logi(__name__, "Exiting loop.")
