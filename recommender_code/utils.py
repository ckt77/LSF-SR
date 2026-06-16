import torch
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import tqdm
import math
import os
import matplotlib.pyplot as plt
from datetime import datetime


def log_and_print(message, log_file=None, print_to_console=True):
    if print_to_console:
        print(message)

    if log_file:
        clean_msg = message.rstrip()
        log_file.write(clean_msg + '\n')
        log_file.flush()


def preprocess_training_sequences(sequence_list, target_list, max_seq_len):
    """
    This function converts all sequences to pre-padded numpy arrays.

    Args:
        sequence_list: List of item sequences (lists)
        target_list: List of target items
        max_seq_len: Maximum sequence length

    Returns:
        tuple: (padded_seqs, targets) - Preprocessed numpy arrays
            - padded_seqs: numpy array of shape (num_samples, max_seq_len) with dtype int32
            - targets: numpy array of shape (num_samples,) with dtype int32
    """

    num_samples = len(sequence_list)
    padded_seqs = np.zeros((num_samples, max_seq_len), dtype=np.int32)
    targets = np.array(target_list, dtype=np.int32)

    for idx, seq in enumerate(sequence_list):
        length = len(seq)

        if length < max_seq_len:
            padded_seqs[idx, -length:] = seq[:]
        else:
            padded_seqs[idx] = np.array(seq[-max_seq_len:], dtype=np.int32)

    return padded_seqs, targets


class TrainDataset(Dataset):
    def __init__(self, sequence_list, target_list, max_seq_len):
        log_and_print(
            f"Preprocessing {len(sequence_list)} training sequences...")
        self.padded_seqs, self.targets = preprocess_training_sequences(
            sequence_list, target_list, max_seq_len)

    def __len__(self):
        return len(self.padded_seqs)

    def __getitem__(self, idx):
        padded_seq = self.padded_seqs[idx]
        target = self.targets[idx]

        return (idx,
                torch.from_numpy(padded_seq).long(),
                torch.tensor(target, dtype=torch.long))


class TestDataset(Dataset):
    def __init__(self, data_dict, max_seq_len):
        self.uid_list, self.sequence_list, self.target_list, self.sequence_length_list = self.data_process(
            data_dict)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, idx):
        uid = self.uid_list[idx]
        seq = self.sequence_list[idx]
        target = self.target_list[idx]
        length = self.sequence_length_list[idx]

        if length < self.max_seq_len:
            padded_seq = np.zeros(self.max_seq_len, dtype=np.int32)
            padded_seq[-length:] = seq[:]

        else:
            padded_seq = np.array(seq[-self.max_seq_len:], dtype=np.int32)

        return torch.tensor(uid, dtype=torch.long), torch.tensor(padded_seq, dtype=torch.long), torch.tensor(
            target, dtype=torch.long), torch.tensor(length, dtype=torch.long)

    def data_process(self, data_dict):
        uid_list, sequence_list, target_list, sequence_length_list = [], [], [], []

        for uid, item_id_seq in data_dict.items():
            if len(item_id_seq) > 1:
                uid_list.append(uid)
                sequence_list.append(item_id_seq[:-1])
                target_list.append(item_id_seq[-1])
                sequence_length_list.append(len(item_id_seq[:-1]))

        return uid_list, sequence_list, target_list, sequence_length_list


def generate_training_samples(data_dict, max_seq_len):
    uid_list, sequence_list, target_list, sequence_length_list = [], [], [], []

    for uid, item_id_seq in data_dict.items():
        seq_start = 0

        for i in range(1, len(item_id_seq)):
            if i - seq_start > max_seq_len:
                seq_start += 1

            uid_list.append(uid)
            sequence_list.append(item_id_seq[seq_start:i])
            target_list.append(item_id_seq[i])
            sequence_length_list.append(i - seq_start)

    return uid_list, sequence_list, target_list, sequence_length_list


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)  # user_id -> list of item_ids
    user_train = {}
    user_valid = {}
    user_test = {}

    # Read interactions from file and track maximum user/item IDs
    with open(fname, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

    for user in User:
        # leave-one-out strategy
        user_train[user] = User[user][:-2]
        user_valid[user] = User[user][:-1]
        user_test[user] = User[user][:]

    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(args, model, eval_data, load_best_model=True, model_file=None):
    """
    Full-prediction evaluation on args.device (same as training, often GPU).
    Use this path to confirm contiguous/register_buffer/cuDNN fixes behave as expected.
    """

    device = torch.device(args.device)

    if load_best_model:
        if model_file:
            checkpoint_file = model_file
        else:
            checkpoint_file = args.saved_model_file

        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        saved_epoch = checkpoint.get('epoch', 'unknown')
        message_output = 'Loading model structure and parameters from {} (epoch {})'.format(
            checkpoint_file, saved_epoch)
        log_and_print(message_output)

    model.to(device)
    model.eval()

    prog_iter = tqdm(eval_data, leave=False)  # progress bar
    scores_list = []
    labels = []

    with torch.no_grad():
        for batch in prog_iter:
            item_seq = batch[1].to(device)
            target_item = batch[2].to(device)
            bs_scores = model.full_sort_predict(item_seq).detach().cpu()
            bs_labels = (target_item - 1).reshape(-1, 1).cpu()
            scores_list.append(bs_scores)
            labels.append(bs_labels)

    scores = torch.cat(scores_list, axis=0)
    _, topk_indices = torch.topk(
        scores, k=20, dim=1, largest=True, sorted=True)
    pred_list = topk_indices.cpu().numpy().tolist()
    labels = torch.cat(labels, axis=0).numpy().tolist()
    result = get_full_sort_score(labels, pred_list)

    return result


def valid_epoch(args, model, valid_data):
    valid_result = evaluate(args, model, valid_data, load_best_model=False)
    valid_score = valid_result[args.valid_metric]

    return valid_score, valid_result


def save_checkpoint(epoch, model, saved_model_file):
    state = {
        'state_dict': model.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, saved_model_file)


def generate_train_loss_output(epoch_idx, s_time, e_time, losses):
    """Format training loss for logging. losses must be (rec_loss, weighted_kl_loss, joint_loss)."""

    des = 4
    rec_loss, weighted_kl_loss, joint_loss = losses
    train_loss_output = (('\n\n[Epoch %d] training') + ' [' + ('time') +
                         ': %.2fs, ') % (epoch_idx, e_time - s_time)
    train_loss_output += f"rec_loss: {rec_loss:.{des}f}, weighted_kl_loss: {weighted_kl_loss:.{des}f}, joint_loss: {joint_loss:.{des}f} | KL/Rec loss ratio = {weighted_kl_loss/rec_loss:.{des}%}]"

    return train_loss_output


def get_full_sort_score(labels, pred_list):
    recall, ndcg, mrr = [], [], []

    for k in [5, 10, 20]:
        recall.append(recall_k(labels, pred_list, k))
        ndcg.append(ndcg_k(labels, pred_list, k))
        mrr.append(mrr_k(labels, pred_list, k))

    result_dic = {
        "HIT@5": round(recall[0], 4), "NDCG@5": round(ndcg[0], 4), "MRR@5": round(mrr[0], 4),
        "HIT@10": round(recall[1], 4), "NDCG@10": round(ndcg[1], 4), "MRR@10": round(mrr[1], 4),
        "HIT@20": round(recall[2], 4), "NDCG@20": round(ndcg[2], 4), "MRR@20": round(mrr[2], 4),
    }

    return result_dic


def recall_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0

    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])

        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1

    return sum_recall / true_users


def ndcg_k(actual, predicted, topk):
    res = 0

    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg

    return res / float(len(actual))


def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])

    if not res:
        return 1.0
    else:
        return res


def mrr_k(actual, predicted, topk):
    sum_mrr = 0.0
    num_users = len(predicted)

    for i in range(num_users):
        act_set = set(actual[i])

        for rank, item in enumerate(predicted[i][:topk], start=1):
            if item in act_set:
                sum_mrr += 1.0 / rank
                break

    return sum_mrr / num_users


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(args, model, train_data, optimizer,
                epoch_idx, log_file=None, show_progress=True):
    model.train()

    total_rec_loss = 0.0
    total_weighted_kl_loss = 0.0
    total_joint_loss = 0.0
    num_batches = 0

    iter_data = (
        tqdm(
            enumerate(train_data),
            total=len(train_data),
            desc=f"Train {epoch_idx:>5}",
        ) if show_progress else enumerate(train_data)
    )

    try:
        for _, batch_data in iter_data:
            _, item_seq, target_items = batch_data
            item_seq = item_seq.to(args.device)
            target_items = target_items.to(args.device)

            optimizer.zero_grad()

            total_rec_loss_with_kl, rec_loss, weighted_kl_loss, _ = model.calculate_loss(
                item_seq, target_items)
            joint_loss = total_rec_loss_with_kl
            rec_loss_val = rec_loss.item()
            weighted_kl_loss_val = weighted_kl_loss.item()
            joint_loss_val = joint_loss.item()
            total_rec_loss += rec_loss_val
            total_weighted_kl_loss += weighted_kl_loss_val
            total_joint_loss += joint_loss_val

            num_batches += 1

            if torch.isnan(joint_loss):
                raise ValueError('Training loss is nan')

            joint_loss.backward()

            # ---------- Gradient Monitoring (every 10 epochs) ---------------#
            if epoch_idx % 10 == 0 and num_batches == 1:  # Only check first batch of every 10th epoch
                total_grad_norm = 0.0
                layer_grad_norms = {}

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                        layer_name = name.split('.')[0]

                        if layer_name not in layer_grad_norms:
                            layer_grad_norms[layer_name] = []

                        layer_grad_norms[layer_name].append(param_norm.item())

                total_grad_norm = total_grad_norm ** (1. / 2)
                grad_info = f"\n[Epoch {epoch_idx}] Gradient Analysis:"
                grad_info += f"\n  Total gradient norm: {total_grad_norm:.4f}"

                for layer_name, norms in sorted(layer_grad_norms.items()):
                    avg_norm = sum(norms) / len(norms)
                    max_norm = max(norms)
                    grad_info += f"\n  - {layer_name}: avg={avg_norm:.4f}, max={max_norm:.4f} ({len(norms)} params)"

                if total_grad_norm < 0.01:
                    grad_info += f"\n Warning: Gradient vanishing detected! (norm < 0.01)"
                elif total_grad_norm > 100:
                    grad_info += f"\n Warning: Gradient explosion detected! (norm > 100)"
                print(grad_info)

                log_and_print(
                    grad_info,
                    log_file=log_file,
                    print_to_console=False)

            if hasattr(args, 'max_grad_norm') and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)

            optimizer.step()

    except (KeyboardInterrupt, EOFError) as e:
        if isinstance(e, EOFError):
            log_and_print(
                "\nDataLoader interrupted (possibly due to Ctrl+C)")

        raise KeyboardInterrupt("Training interrupted") from e

    # Calculate average losses across all batches in the epoch
    if num_batches > 0:
        avg_rec_loss = total_rec_loss / num_batches
        avg_weighted_kl_loss = total_weighted_kl_loss / num_batches
        avg_joint_loss = total_joint_loss / num_batches

    else:
        avg_rec_loss = 0.0
        avg_weighted_kl_loss = 0.0
        avg_joint_loss = 0.0

    return (avg_rec_loss, avg_weighted_kl_loss, avg_joint_loss)


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError(f'Not a valid boolean string: {s}')


def calculate_beta(epoch, args):
    """
    Calculate KL loss coefficient (beta in beta-CVAE) based on annealing strategy.

    Args:
        epoch: Current training epoch (0-indexed)
        args: Arguments containing annealing parameters

    Returns:
        float: beta (KL loss coefficient) for current epoch
    """

    if not args.use_kl_annealing:
        return args.beta

    if args.kl_anneal_strategy == 'linear':
        if epoch < args.kl_anneal_epochs:
            progress = epoch / args.kl_anneal_epochs

            return args.beta_start + \
                (args.beta_end - args.beta_start) * progress

        else:
            return args.beta_end

    elif args.kl_anneal_strategy == 'cyclical':
        cyclical_ratio = max(0.0, min(1.0, args.cyclical_ratio))
        anneal_epochs = int(args.cyclical_period * cyclical_ratio)
        position_in_cycle = epoch % args.cyclical_period

        if position_in_cycle < anneal_epochs:
            progress = position_in_cycle / anneal_epochs
            return args.beta_start + \
                (args.beta_end - args.beta_start) * progress

        else:
            return args.beta_end

    elif args.kl_anneal_strategy == 'sigmoid':
        if epoch < args.kl_anneal_epochs:
            progress = epoch / args.kl_anneal_epochs
            sigmoid_k = getattr(args, 'sigmoid_k', 5.0)
            sigmoid_input = sigmoid_k * (progress - 0.5) * 2
            sigmoid_value = 1.0 / (1.0 + math.exp(-sigmoid_input))

            return args.beta_start + \
                (args.beta_end - args.beta_start) * sigmoid_value

        else:
            return args.beta_end

    else:
        raise ValueError(
            f"Invalid KL annealing strategy: {args.kl_anneal_strategy}")


def load_item_semantic_embeddings(args):
    item_semantic_emb = torch.load(args.item_semantic_emb_file)
    # Force padding (index 0) to be a zero vector for the padding item
    item_semantic_emb[0] = 0.0
    args.item_semantic_emb = item_semantic_emb.to(args.device)

    log_and_print("Item semantic embeddings loaded.")


def plot_valid_metric_trend(
        epoch_metric_pairs, valid_metric, experiment_run_dir, current_time, log_file=None):
    """
    Plot validation metric trend from per-epoch recorded values.

    Args:
        epoch_metric_pairs: list of (epoch_idx, metric_value) recorded during training
        valid_metric: metric name used for validation, e.g. 'NDCG@20' (used for label/title/filename)
        experiment_run_dir: directory to save the plot
        current_time: timestamp string for filename
        log_file: optional file handle for logging

    Returns:
        path to saved plot file, or None if no data or on error
    """

    try:
        if not epoch_metric_pairs:
            log_and_print(
                "Warning: No validation metric history to plot, skipping", log_file=log_file)
            return None

        epoch_metric_pairs = sorted(epoch_metric_pairs, key=lambda x: x[0])
        epochs = [p[0] for p in epoch_metric_pairs]
        values = [p[1] for p in epoch_metric_pairs]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, values, marker='o',
                 linestyle='-', linewidth=2, markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(valid_metric, fontsize=12)
        plt.title(f'Validation {valid_metric} Trend',
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        safe_metric_name = valid_metric.replace('@', '_')
        plot_filename = os.path.join(
            experiment_run_dir, f'{safe_metric_name}_trend_{current_time}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

        log_and_print(
            f"{valid_metric} trend plot saved to: {plot_filename}", log_file=log_file)

        return plot_filename

    except Exception as e:
        log_and_print(
            f"Error plotting {valid_metric} trend: {e}", log_file=log_file)
        
        return None


def setup_experiment_directory(args):
    """
    Create experiment directory structure and save configuration.

    Directory structure:
        experiments/
          └── {dataset_name}/
              └── {experiment_name}/
                  └── {timestamp}/

    Args:
        args: Arguments object with experiment_name and dataset_file

    Returns:
        tuple: (experiment_run_dir, current_time) - Directory path and timestamp string
    """

    # Extract dataset name from dataset_file path
    dataset_file = args.dataset_file
    dataset_basename = os.path.basename(dataset_file)
    dataset_name = os.path.splitext(dataset_basename)[0]

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = os.path.join('experiments', dataset_name)
    named_experiment_dir = os.path.join(dataset_dir, args.experiment_name)
    experiment_run_dir = os.path.join(named_experiment_dir, current_time)

    # Create directories if they don't exist
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.isdir(named_experiment_dir):
        os.makedirs(named_experiment_dir)
    if not os.path.isdir(experiment_run_dir):
        os.makedirs(experiment_run_dir)

    # Save configuration
    with open(os.path.join(experiment_run_dir, 'config.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v)
                for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # Set file paths in args
    args.saved_model_file = os.path.join(
        experiment_run_dir, f"model_{current_time}.pth")
    args.saved_result_file = os.path.join(
        experiment_run_dir, f"results_{current_time}.txt")
    args.log_file_path = os.path.join(
        experiment_run_dir, f"training_{current_time}.log")

    return experiment_run_dir, current_time


def initialize_training_log(args, log_file):
    messages = []
    messages.append('=' * 80)
    messages.append(f'Experiment: {args.experiment_name}')
    messages.append(f'Model save path: {args.saved_model_file}')
    messages.append(f'Result save path: {args.saved_result_file}')

    if getattr(args, 'lr_use_cosine_annealing', False):
        messages.append(
            f'Learning rate: {args.learning_rate} (initial), Cosine Annealing with Warm Restarts (period={args.lr_cosine_period}, eta_min={args.lr_cosine_eta_min})')

    else:
        messages.append(
            f'Learning rate: {args.learning_rate} (fixed)')

    if args.use_kl_annealing:
        if args.kl_anneal_strategy == 'linear':
            messages.append(
                f'KL Annealing: {args.kl_anneal_strategy} - from {args.beta_start} to {args.beta_end} (over {args.kl_anneal_epochs} epochs)')

        elif args.kl_anneal_strategy == 'cyclical':
            messages.append(
                f'KL Annealing: {args.kl_anneal_strategy} - period={args.cyclical_period}, ratio={args.cyclical_ratio}, range=[{args.beta_start}, {args.beta_end}]')

        elif args.kl_anneal_strategy == 'sigmoid':
            sigmoid_k = getattr(args, 'sigmoid_k', 5.0)
            messages.append(
                f'KL Annealing: {args.kl_anneal_strategy} - from {args.beta_start} to {args.beta_end} (over {args.kl_anneal_epochs} epochs, k={sigmoid_k})')

    else:
        messages.append(f'beta (KL loss coefficient): {args.beta} (fixed)')

    messages.append(f'Training epochs: {args.epochs}')
    flow_type = getattr(args, 'flow_type', 'planar')
    messages.append(
        f'Posterior flow: {flow_type} (num_flows={getattr(args, "num_flows", 2)})')
    messages.append('=' * 80 + '\n')
    for msg in messages:
        log_and_print(msg, log_file=log_file)
