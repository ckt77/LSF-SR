from time import time
import torch
import argparse
from model import TransformerRec
from utils import *
from recbole.utils import early_stopping, dict2str
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import traceback

parser = argparse.ArgumentParser()
# Config arguments in alphabetical order (by option name)
parser.add_argument('--attn_dropout_prob', default=0.3, type=float)
parser.add_argument('--beta', default=0.01, type=float,
                    help='Coefficient for KL loss in beta-CVAE (fixed when annealing is off)')
parser.add_argument('--beta_end', default=0.01, type=float,
                    help='KL loss coefficient at end of annealing')
parser.add_argument('--beta_start', default=0.001, type=float,
                    help='KL loss coefficient at start of annealing')
parser.add_argument('--cyclical_period', default=30, type=int,
                    help='Cyclical annealing period length (epochs)')
parser.add_argument('--cyclical_ratio', default=0.8, type=float,
                    help='Cyclical annealing growth phase ratio (0-1)')
parser.add_argument('--dataset_file',
                    default='../build_datasets_and_prompts/data/Beauty/Beauty.txt',
                    type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--epochs', default=1000, type=int,
                    help='Number of training epochs')
parser.add_argument('--eval_step', default=1, type=int)
parser.add_argument('--experiment_name',
                    default='Cyclical', type=str, help='Experiment name')
parser.add_argument('--flow_init_scale', default=0.001, type=float,
                    help='Init scale for flow parameters (smaller = closer to identity at start)')
parser.add_argument('--flow_type', default='planar', type=str,
                    choices=['planar', 'radial'],
                    help='Posterior flow type: planar or radial')
parser.add_argument('--hidden_act', default='gelu', type=str)
parser.add_argument('--hidden_dropout_prob', default=0.3, type=float)
parser.add_argument('--hidden_size', default=64, type=int)
parser.add_argument('--initializer_range', default=0.02, type=float)
parser.add_argument('--inner_size', default=256, type=int,
                    help='Inner size for TransformerEncoder')
parser.add_argument('--item_semantic_emb_file',
                    default='../build_datasets_and_prompts/data/Beauty/beauty_item_semantic_embeddings.pt',
                    type=str)
parser.add_argument('--kl_anneal_epochs', default=100, type=int,
                    help='Number of epochs for KL annealing (linear/sigmoid)')
parser.add_argument('--kl_anneal_strategy', default='cyclical', type=str,
                    choices=['linear', 'cyclical', 'sigmoid'],
                    help='KL annealing strategy')
parser.add_argument('--latent_dim', default=32, type=int,
                    help='Latent dimension for CVAE fusion module')
parser.add_argument('--layer_norm_eps', default=1e-12, type=float)
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    help='Initial learning rate (applied to all parameters)')
parser.add_argument('--lr_cosine_eta_min', default=1e-4, type=float,
                    help='Minimum learning rate in Cosine Annealing')
parser.add_argument('--lr_cosine_period', default=30, type=int,
                    help='Period length (epochs) for Cosine Annealing restarts')
parser.add_argument('--lr_use_cosine_annealing', default=True, type=str2bool,
                    help='Whether to use Cosine Annealing with Warm Restarts for learning rate')
parser.add_argument('--max_grad_norm', default=1.0, type=float,
                    help='Maximum gradient norm for clipping (0 to disable)')
parser.add_argument('--max_seq_len', default=20, type=int,
                    help='Maximum sequence length')
parser.add_argument('--num_heads', default=2, type=int,
                    help='Number of attention heads')
parser.add_argument('--num_layers', default=2, type=int,
                    help='Number of Transformer layers')
parser.add_argument('--num_flows', default=2, type=int,
                    help='Number of Normalizing Flow layers in posterior (0 to disable)')
parser.add_argument('--seed', default=2026, type=int)
parser.add_argument('--sigmoid_k', default=5.0, type=float,
                    help='Sigmoid annealing steepness parameter (higher = steeper curve)')
parser.add_argument('--stopping_step', default=60, type=int)
parser.add_argument('--test_batch_size', default=1024, type=int)
parser.add_argument('--train_batch_size', default=256, type=int)
parser.add_argument('--use_kl_annealing', default=True,
                    type=str2bool, help='Whether to use KL annealing')
parser.add_argument('--valid_metric', default='NDCG@20', type=str)
parser.add_argument('--weight_decay', default=0.0, type=float)

args = parser.parse_args()


if __name__ == '__main__':
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    experiment_run_dir, current_time = setup_experiment_directory(args)

    with open(args.log_file_path, 'w') as log_file:
        initialize_training_log(args, log_file)

        try:
            log_and_print(
                "\n[Stage 1] Loading item semantic embeddings...", log_file=log_file)
            load_item_semantic_embeddings(args)

            log_and_print("\n[Stage 2] Data processing...", log_file=log_file)
            [user_train, user_valid, user_test, usernum,
                itemnum] = data_partition(args.dataset_file)
            args.user_num = usernum
            args.item_num = itemnum
            uid_list, sequence_list, target_list, _ = generate_training_samples(
                user_train, args.max_seq_len)
            generate_training_samples(user_valid, args.max_seq_len)
            generate_training_samples(user_test, args.max_seq_len)

            log_and_print(
                f"Training sequences: {len(sequence_list)}",
                log_file=log_file)
            log_and_print(f"\nUsers: {usernum}", log_file=log_file)
            log_and_print(f"Items: {itemnum}", log_file=log_file)

            log_and_print(
                "\n[Stage 3] Creating data loaders...", log_file=log_file)
            TrainData = TrainDataset(
                sequence_list, target_list, args.max_seq_len)
            use_cuda = args.device == 'cuda'
            TrainDataLoader = DataLoader(
                TrainData,
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=use_cuda
            )

            ValData = TestDataset(user_valid, args.max_seq_len)
            ValDataLoader = DataLoader(
                ValData,
                batch_size=args.test_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=use_cuda
            )

            TestData = TestDataset(user_test, args.max_seq_len)
            TestDataLoader = DataLoader(
                TestData,
                batch_size=args.test_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=use_cuda
            )

            log_and_print(
                "\n[Stage 4] Creating model...", log_file=log_file)
            model = TransformerRec(args).to(args.device)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel()
                                   for p in model.parameters() if p.requires_grad)
            log_and_print(
                f"Total parameters: {total_params:,}", log_file=log_file)
            log_and_print(
                f"Trainable parameters: {trainable_params:,}", log_file=log_file)

            log_and_print(
                "\n[Stage 5] Setting up optimizer...", log_file=log_file)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )

            # Setup learning rate scheduler
            scheduler = None
            if args.lr_use_cosine_annealing:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=args.lr_cosine_period,
                    T_mult=1,
                    eta_min=args.lr_cosine_eta_min
                )

                log_and_print(
                    f"Cosine Annealing Warm Restarts: period={args.lr_cosine_period}, eta_min={args.lr_cosine_eta_min}",
                    log_file=log_file)

            else:
                log_and_print(
                    f"Using fixed learning rate: {args.learning_rate}", log_file=log_file)

            log_and_print("\n[Stage 6] Starting training...",
                          log_file=log_file)
            start_epoch = 0
            epoch_valid_metric_history = []
            best_valid_score = -np.inf
            cur_step = 0
            best_valid_result = None
            best_epoch = -1

            for epoch_idx in range(start_epoch, args.epochs):
                if scheduler is not None and epoch_idx > 0:
                    scheduler.step()

                current_beta = calculate_beta(epoch_idx, args)
                model.set_beta(current_beta)

                training_start_time = time()
                train_loss = train_epoch(
                    args, model, TrainDataLoader, optimizer, epoch_idx, log_file=log_file)
                training_end_time = time()
                train_loss_output = generate_train_loss_output(
                    epoch_idx, training_start_time, training_end_time, train_loss)

                current_lr = optimizer.param_groups[0]['lr']
                lr_info = f" | lr={current_lr:.2e}"

                if args.use_kl_annealing and args.kl_anneal_strategy == 'cyclical':
                    cycle_num = epoch_idx // args.cyclical_period
                    train_loss_output += f" | CVAE (beta={current_beta:.5f}, cycle={cycle_num}){lr_info}"
                else:
                    train_loss_output += f" | CVAE (beta={current_beta:.5f}){lr_info}"

                log_and_print(train_loss_output, log_file=log_file)

                if (epoch_idx) % args.eval_step == 0:
                    valid_start_time = time()
                    valid_score, valid_result = valid_epoch(
                        args, model, ValDataLoader)
                    epoch_valid_metric_history.append((epoch_idx, valid_score))
                    should_count_early_stopping = True

                    if args.use_kl_annealing and args.kl_anneal_strategy in [
                            'linear', 'sigmoid']:
                        if epoch_idx < args.kl_anneal_epochs:
                            should_count_early_stopping = False
                            cur_step = 0

                            if valid_score > best_valid_score:
                                best_valid_score = valid_score
                                update_flag = True
                            else:
                                update_flag = False

                            stop_flag = False

                        elif epoch_idx == args.kl_anneal_epochs:
                            cur_step = 0
                            early_stop_msg = f"[Early Stopping] Starting early stopping count from epoch {epoch_idx} (after {args.kl_anneal_strategy} annealing completed). Will stop if no improvement for {args.stopping_step} epochs."
                            log_and_print(early_stop_msg, log_file=log_file)
                            should_count_early_stopping = True

                        else:
                            should_count_early_stopping = True

                    if should_count_early_stopping:
                        best_valid_score, cur_step, stop_flag, update_flag = early_stopping(
                            valid_score,
                            best_valid_score,
                            cur_step,
                            max_step=args.stopping_step,
                            bigger=True
                        )

                    valid_end_time = time()
                    valid_result_output = (
                        'valid result') + ': \n' + dict2str(valid_result)
                    log_and_print(valid_result_output, log_file=log_file)

                    if update_flag:
                        best_epoch = epoch_idx
                        save_checkpoint(epoch_idx, model,
                                        args.saved_model_file)
                        update_output = (
                            'Saving current best') + ': %s (epoch %d)' % (args.saved_model_file, epoch_idx)
                        log_and_print(update_output, log_file=log_file)
                        best_valid_result = valid_result

                    if stop_flag:
                        stop_output = 'Finished training, best eval result in epoch %d' % best_epoch
                        log_and_print(stop_output, log_file=log_file)
                        break

            log_and_print("\n[Stage 7] Testing...", log_file=log_file)
            test_result = evaluate(args, model, TestDataLoader, load_best_model=True,
                                   model_file=args.saved_model_file)
            df = pd.DataFrame([test_result])
            pd.set_option('display.float_format', lambda x: '%.4f' % x)
            df.to_csv(args.saved_result_file, index=False, sep='\t')
            test_result_output = (
                '\ntest result') + f': {dict2str(test_result) if test_result else "No result"}'
            log_and_print(test_result_output, log_file=log_file)
            log_and_print(
                f"Test results saved to: {args.saved_result_file}", log_file=log_file)

            if best_valid_result:
                best_valid_output = f"\nBest validation result: {dict2str(best_valid_result)}"
                log_and_print(best_valid_output, log_file=log_file)

            log_and_print(
                f"\n[Stage 8] Plotting {args.valid_metric} trend...", log_file=log_file)
            plot_valid_metric_trend(epoch_valid_metric_history, args.valid_metric,
                                    experiment_run_dir, current_time, log_file=log_file)

            log_and_print("\nTraining completed!", log_file=log_file)

        except (KeyboardInterrupt, EOFError) as e:
            if isinstance(e, EOFError):
                log_and_print(
                    "\nTraining interrupted (EOFError from DataLoader - likely due to Ctrl+C)", log_file=log_file)

            else:
                log_and_print(
                    "\nTraining interrupted by user (KeyboardInterrupt)", log_file=log_file)

        except Exception as e:
            log_and_print(f"\nError during training: {e}", log_file=log_file)
            traceback.print_exc()
            raise

        finally:
            log_and_print("\nPerforming final cleanup...", log_file=log_file)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_and_print("GPU cache cleared.", log_file=log_file)
