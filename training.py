import os
import sys
import numpy as np
import torch.backends
import torch.nn as nn
import time
from time import gmtime, strftime
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from datetime import datetime
from manual_dataset import *
from rich.progress import Progress
from models import ResNet, SELDNet
from utilities import *
import argparse
from inference import inference


def test_epoch(data_generator,  model, criterion, device, nb_batches=1000):
    nb_batches_processed, validation_loss = 0, 0.
    model.eval()

    # Create an instance of the aggregator
    seld_metric = SELDMetricsAzimuth(n_classes=3, azimuth_threshold=20, sed_threshold=0.5)

    with Progress(transient=True) as progress:
        task = progress.add_task("[green]Validation: ", total=nb_batches)

        with torch.no_grad():
            for data, target in data_generator:

                data , target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                validation_loss += loss.item()
                nb_batches_processed += 1

                # Update aggregator with this batch’s ground truth + predictions
                seld_metric.update(gt=convert_output(target), pred=convert_output(output))

                progress.update(task, advance=1)

    validation_loss /= nb_batches_processed
    # Compute the final metrics across all batches
    ER, F, LE, LR = seld_metric.compute()

    del data, target, output
    torch.cuda.empty_cache()
    return validation_loss, ER, F, LE, LR


def train_epoch(data_generator, optimizer, model, criterion, device, 
                nb_batches=1000, step_scheduler=None, batch_scheduler=None):
    nb_train_batches, train_loss = 0, 0.
    model.train()

    with Progress(transient=True) as progress:
        task = progress.add_task("[red]Training: ", total=nb_batches)

        for data, target in data_generator:
            data , target = data.to(device), target.to(device)

            # Training step
            output = model(data)
            loss = criterion(output, target)

            # Clear gradients
            optimizer.zero_grad(set_to_none=True)

            # Backprop
            loss.backward()

            # Update parameters
            optimizer.step()

            train_loss += loss.item()
            nb_train_batches += 1
            progress.update(task, advance=1)

            # For step based schedulers
            if step_scheduler is not None: step_scheduler.step()

    train_loss /= nb_train_batches

    # For epoch based schedulers
    if batch_scheduler is not None: batch_scheduler.step()

    del data, target, output
    torch.cuda.empty_cache()
    return train_loss


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training Script with Argparse Integration")

    # Experiment Identification
    parser.add_argument("--unique_name", type=str, default="demo", help="Unique name for the experiment/run.")

    # Model Configurations
    parser.add_argument("--use_dsc", action="store_true", help="Use Depthwise Separable Convolutions")
    parser.add_argument("--use_btndsc", action="store_true",
                        help="Use Bottleneck ResBlks. Depthwise Separable Convolutions enabled by default.")
    parser.add_argument("--lightweight", action="store_true", help="Use the lightweight version of the models")

    # Directories
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save log files.")
    parser.add_argument("--model_dir", type=str, default="./model_weights", help="Directory to save model weights.")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 regularization) factor.")
    parser.add_argument("--n_workers", type=int, default=0, help="Number of worker threads for data loading.")
    parser.add_argument("--sched", type=str, default="batch", help="Type of learning rate scheduler used (batch/step).")
    parser.add_argument("--feat_label_dir", type=str, default="./feat_label",
                        help="Directory where all the features and labels are stored.")
    parser.add_argument("--model", type=str, default="resnet", help="Model Choice")

    # Data Augmentation
    parser.add_argument("--use_augmentations", action="store_true", help="Enable data augmentations during training.")

    return parser.parse_args()


def main():
    args = parse_arguments()
    unique_name = args.unique_name

    # Create necessary directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # Device configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.autograd.set_detect_anomaly(True)

    # Create the logging file
    run_starttime = datetime.now().strftime("%d%m%y_%H%M")
    log_file = os.path.join("./logs", "{}_{}_logs.txt".format(run_starttime, unique_name))
    logger = open(log_file, "w")

    # Training setup
    nb_epoch = args.epochs
    batch_size = args.batch_size
    n_workers = args.n_workers

    # Unique name for the run
    model_name = os.path.join(args.model_dir,'{}_{}_model.h5'.format(run_starttime, unique_name))

    # Logging
    write_and_print(logger, f"Project: {run_starttime}_{args.unique_name}")
    write_and_print(logger, f"Training started: {datetime.now().strftime('%d%m%y_%H%M%S')}")
    write_and_print(logger, f"No. of Epochs: {args.epochs}")
    write_and_print(logger, f"Unique Name: {run_starttime}_{args.unique_name}")
    write_and_print(logger, f"Device used: {device}")
    write_and_print(logger, f"Augmentations used: {args.use_augmentations}")

    # Load dataset splits
    dataset = seldDatabase(feat_label_dir=args.feat_label_dir)
    train_data = dataset.get_split("train")
    test_data = dataset.get_split("test")
    test_batch_size = test_data["test_batch_size"]

    # Set up data augmentation if used
    use_augmentations = args.use_augmentations
    if use_augmentations:
        training_transforms = ComposeTransformNp([
            RandomShiftUpDownNp(freq_shift_range=10),
            CompositeCutout(image_aspect_ratio=80/191,
                            n_zero_channels=3),
        ])
    else:
        training_transforms = None

    train_dataset = seldDataset(db_data=train_data, transform=training_transforms)
    test_dataset = seldDataset(db_data=test_data)

    # Input/Output shapes
    sample_x, sample_y = train_dataset[0]
    data_in = sample_x.shape
    data_out = sample_y.shape
    write_and_print(logger, f"FEATURES:\n\tdata_in: {data_in}\n\tdata_out: {data_out}")


    # Creating the dataloaders
    training_dataloader = DataLoader(dataset=train_dataset,
                                     batch_size=batch_size, shuffle=True,
                                     num_workers=n_workers, drop_last=False,
                                     pin_memory=True, prefetch_factor=2)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=test_batch_size, shuffle=False,
                                 num_workers=n_workers, drop_last=False,
                                 pin_memory=True, prefetch_factor=2)

    n_batches = len(training_dataloader)
    n_val_batches = len(test_dataloader)
    write_and_print(logger, f"Training dataloader: {n_batches} batches (batch size: {args.batch_size})")

    # Deciding on model architecture
    if "resnet" in args.model.lower():
        fps = 2 if "2fps" in args.feat_label_dir else 10
        model = ResNet(in_feat_shape=data_in,
                       out_feat_shape=data_out,
                       use_dsc=args.use_dsc, btn_dsc=args.use_btndsc,
                       lightweight=args.lightweight, fps=fps).to(device)
        write_and_print(logger, "Using ResNet-GRU!")
        write_and_print(logger, f"BTNDSC:      {args.use_btndsc}")
        write_and_print(logger, f"DSC:         {args.use_dsc}")
        write_and_print(logger, f"Lightweight: {args.lightweight}")

    else:
        model = SELDNet(in_feat_shape=data_in,
                        out_feat_shape=data_out).to(device)
        write_and_print(logger, "Using SELDNet for training")
    write_and_print(logger, f"Number of params: {count_parameters(model)/1e6:.3f}M")

    # Adam Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Now we set the learning rate scheduler
    num_batches_per_epoch = len(training_dataloader)
    total_steps = nb_epoch * num_batches_per_epoch

    # Decide scheduler based on argument
    step_scheduler = None
    batch_scheduler = None

    if args.sched == "step":
        step_scheduler = CosineWarmup_StepScheduler(optimizer, total_steps=total_steps)
        print("Cosine Annealing w/ Warmup Step Scheduler is used!\nTotal Number of Steps : {}".format(total_steps))
    elif args.sched == "batch":
        batch_scheduler = CustomTriPhaseScheduler(optimizer, total_epochs=args.epochs, peak_lr=args.learning_rate)
        print("Custom Tri-Phase Scheduler used!")
    else:
        batch_scheduler = DecayScheduler(optimizer)
        print("Batch Decay Scheduler used!")

    # Defining the loss function to be used, which is dependent on our output format
    criterion = nn.MSELoss()

    try:
        best_er, best_f1, best_le, best_lr, best_seld, best_epoch = 9999, 0., 180., 0., 9999, 0

        for epoch_cnt in range(nb_epoch):
            # ---------------------------------------------------------------------
            # TRAINING
            # ---------------------------------------------------------------------
            start_time = time.time()
            train_loss = train_epoch(training_dataloader, optimizer, model, criterion, device, 
                                     nb_batches = n_batches, step_scheduler=step_scheduler, batch_scheduler=batch_scheduler)
            train_time = time.time() - start_time

            # ---------------------------------------------------------------------
            # VALIDATION
            # ---------------------------------------------------------------------
            start_time = time.time()
            val_loss, ER, F, LE, LR = test_epoch(test_dataloader, model, criterion, device, nb_batches = n_val_batches)
            seld_err = (ER + (1-F) + LE/180 + (1-LR))/4
            if seld_err < best_seld:
                torch.save(model.state_dict(), model_name)
                best_er, best_f1, best_le, best_lr, best_seld = ER, F, LE, LR, seld_err
                best_epoch = epoch_cnt
            val_time = time.time() - start_time

            # Get the current learning rate from the optimizer
            current_lr = optimizer.param_groups[0]['lr']

            # ---------------------------------------------------------------------
            # LOGGING METRICS AND VARIABLES
            # ---------------------------------------------------------------------
            write_and_print(logger, 
                'epoch: {}, time: {:0.2f}/{:0.2f}, '
                'train_loss: {:0.4f}, val_loss: {:0.4f}, '
                'ER/F1/LE/LR/SELD: {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.3f}, '
                'LR: {:.6f}'.format(
                    epoch_cnt, train_time, val_time,
                    train_loss, val_loss,
                    ER, F, LE, LR, seld_err,
                    current_lr)
            )

    except KeyboardInterrupt:
        write_and_print(logger, "Training ended prematurely.")

    # Getting some classwise stats
    write_and_print(logger, "Best model saved at: {}".format(model_name))
    model.load_state_dict(torch.load(model_name, map_location='cpu'))
    er, f1, le, lr = inference(test_dataloader, 
                                model=model, 
                                device=device, 
                                nb_batches=len(test_dataloader))
    e_seld = (er + (1-f1) + (le/180) + (1-lr))/4
    print("Best ER/F1/LE/LR/SELD: {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.3f}".format(er, f1, le, lr, e_seld))

    # Final Output Stats
    write_and_print(logger, 
                'Best Epoch: {}, ER/F1/LE/LR/SELD: {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.3f}'.format(
                    best_epoch,
                    best_er, best_f1, best_le, best_lr, best_seld)
            )


if __name__ == "__main__":
    # Record the start time
    start_time = time.time()

    try:
        # Execute the main function
        sys.exit(main())
    except (ValueError, IOError) as e:
        # Handle exceptions and exit with the error
        sys.exit(e)
    finally:
        # Record the end time
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time

        # Convert elapsed_time to a human-readable format
        hours = int(elapsed_time // 3600)
        remaining_time = elapsed_time % 3600
        minutes = int(remaining_time // 60)
        seconds = remaining_time % 60
        print(f"Execution time: {hours}h {minutes}min {seconds:.2f}s")