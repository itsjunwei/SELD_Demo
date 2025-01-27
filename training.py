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
from models import ResNet
from torchvision import transforms
from utilities import *
import argparse


def test_epoch(data_generator,  model, criterion, device, nb_batches=1000):
    nb_train_batches, validation_loss = 0, 0.
    model.eval()
    
    # Create an instance of the aggregator
    seld_metric = SELDMetricsAzimuth(n_classes=3, azimuth_threshold=20, sed_threshold=0.5)

    with Progress(transient=True) as progress:
        task = progress.add_task("[green]Validation : ", total=nb_batches)

        with torch.no_grad():
            for data, target in data_generator:

                data , target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                validation_loss += loss.item()
                nb_train_batches += 1

                # Update aggregator with this batch’s ground truth + predictions
                target = convert_output(target)
                output = convert_output(output)

                seld_metric.update(gt=target, pred=output)

                progress.update(task, advance=1)

    validation_loss /= nb_train_batches
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
        task = progress.add_task("[red]Training : ", total=nb_batches)

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
            if step_scheduler is not None : step_scheduler.step()

    train_loss /= nb_train_batches

    # For epoch based schedulers
    if batch_scheduler is not None : batch_scheduler.step()

    del data, target, output
    torch.cuda.empty_cache()
    return train_loss


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training Script with Argparse Integration")

    # Experiment Identification
    parser.add_argument(
        "--unique_name",
        type=str,
        default="demo",
        help="Unique name for the experiment/run."
    )

    # Model Configurations
    parser.add_argument(
        "--use_dsc",
        action="store_true",
        help="Use Depthwise Separable Convolutions"
    )

    parser.add_argument(
        "--use_btndsc",
        action="store_true",
        help="Use Bottleneck ResBlks. Depthwise Separable Convolutions enabled by default."
    )

    # Directories
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory to save log files."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./model_weights",
        help="Directory to save model weights."
    )

    # Training Hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization) factor."
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=0,
        help="Number of worker threads for data loading."
    )
    parser.add_argument(
        "--sched",
        type=str,
        default="batch",
        help="Type of learning rate scheduler used (batch/step)."
    )
    parser.add_argument(
        "--feat_label_dir",
        type=str,
        default="./feat_label",
        help="Directory where all the features and labels are stored."
    )

    # Data Augmentation
    parser.add_argument(
        "--use_augmentations",
        action="store_true",
        help="Enable data augmentations during training."
    )

    return parser.parse_args()


def main():
    
    args = parse_arguments()
    unique_name = args.unique_name

    # Make the log and model directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # Device configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.autograd.set_detect_anomaly(True)

    run_starttime = datetime.now().strftime("%d%m%y_%H%M")

    # Create the logging file
    log_file = os.path.join("./logs", "{}_{}_logs.txt".format(run_starttime, unique_name))
    logger = open(log_file, "w")

    # Starting up the wandb logger
    project_title = f"{run_starttime}_{args.unique_name}"
    write_and_print(logger, project_title)


    # Training setup
    nb_epoch = args.epochs
    batch_size = args.batch_size
    n_workers = args.n_workers


    # Unique name for the run
    model_name = os.path.join(args.model_dir,'{}_{}_model.h5'.format(run_starttime, unique_name))
    write_and_print(logger, out_string="Unique Name: {}_{}".format(run_starttime, unique_name))
    write_and_print(logger, out_string="Training started : {}".format(datetime.now().strftime("%d%m%y_%H%M%S")))
    
    dataset = seldDatabase(feat_label_dir=args.feat_label_dir)
    train_data = dataset.get_split("train")
    test_data = dataset.get_split("test")
    test_batch_size = test_data["test_batch_size"]
    
    use_augmentations = args.use_augmentations
    if use_augmentations:
        training_transforms = ComposeTransformNp([
            RandomShiftUpDownNp(freq_shift_range=10),
        ])
    else:
        training_transforms = None

    train_dataset = seldDataset(db_data=train_data, transform=training_transforms)
    test_dataset = seldDataset(db_data=test_data)
    
    sample_x, sample_y = train_dataset[0]
    data_in = sample_x.shape
    data_out = sample_y.shape
    print("In shape: {}, Out shape: {}".format(data_in, data_out))


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
    print("Manual training dataloader created with {} batches using batch size of {}!".format(n_batches, batch_size))

    # Deciding on model architecture
    model = ResNet(in_feat_shape=data_in,
                   out_feat_shape=data_out,
                   use_dsc=args.use_dsc, btn_dsc=args.use_btndsc).to(device)
    print("Using ResNet-GRU!")

    write_and_print(logger, 'FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out)) # Get input and output shape sizes
    print("Number of params : {:.3f}M".format(count_parameters(model)/(10**6)))


    # # Define weight decay settings (exclude bias and normalization layers)
    # print("Training the model for a total of : {} epochs.".format(nb_epoch))
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {
    #         'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         'weight_decay': 1e-4
    #     },
    #     {
    #         'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #         'weight_decay': 0.0
    #     }
    # ]

    # optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # print("Using Adam with Weight Decay optimizer")

    # Adam Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Now we set the learning rate scheduler
    num_batches_per_epoch = len(training_dataloader)
    total_steps = nb_epoch * num_batches_per_epoch
    
    step_scheduler = None
    batch_scheduler = None

    if args.sched == "step":
        step_scheduler = CosineWarmup_StepScheduler(optimizer, total_steps=total_steps)
        print("Cosine Annealing w/ Warmup Step Scheduler is used!\nTotal Number of Steps : {}".format(total_steps))
    elif args.sched == "batch":
        batch_scheduler = DecayScheduler(optimizer, min_lr=args.learning_rate)
        print("Batch Decay Scheduler used!")
    else:
        batch_scheduler = DecayScheduler(optimizer, min_lr=args.learning_rate)
        print("Batch Decay Scheduler used!")

    optimizer.zero_grad()
    optimizer.step()

    # Defining the loss function to be used, which is dependent on our output format
    criterion = nn.MSELoss()

    # Misc. print statements for viewing the training configurations
    write_and_print(logger, "Device used : {}".format(device))
    write_and_print(logger, "Augmentations used : {}".format(use_augmentations))


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

            # ---------------------------------------------------------------------
            # LOGGING METRICS AND VARIABLES
            # ---------------------------------------------------------------------
            # Print stats
            write_and_print(logger, 
                'epoch: {}, time: {:0.2f}/{:0.2f}, '
                'train_loss: {:0.4f}, val_loss: {:0.4f}, '
                'ER/F1/LE/LR/SELD: {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.3f}'.format(
                    epoch_cnt, train_time, val_time,
                    train_loss, val_loss,
                    ER, F, LE, LR, seld_err)
            )

    except KeyboardInterrupt:
        write_and_print(logger, "Training ended prematurely.")

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
        # One minute or more: display in minutes and seconds
        hours = int(elapsed_time // 3600)
        remaining_time = elapsed_time % 3600
        minutes = int(remaining_time // 60)
        seconds = remaining_time % 60
        print(f"Execution time: {hours}h {minutes}min {seconds:.2f}s")
