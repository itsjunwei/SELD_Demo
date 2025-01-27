import numpy as np
import torch.backends
import torch
from torch.utils.data import DataLoader
from manual_dataset import *
from rich.progress import Progress
from models import ResNet
from utilities import *


def inference(data_generator,  model, device, nb_batches=1000, sed_threshold=0.5):
    model.eval()
    
    # Create an instance of the aggregator
    seld_metric = SELDMetricsAzimuth(n_classes=3, azimuth_threshold=20, sed_threshold=sed_threshold, out_class=True)

    with Progress(transient=True) as progress:
        task = progress.add_task("[green]Validation : ", total=nb_batches)

        with torch.no_grad():
            for data, target in data_generator:

                data , target = data.to(device), target.to(device)
                
                # Split the data into managable chunks, in this case 32 items
                data_chunks = torch.split(data, 32)
                output_chunks = []
                for data_chunk in data_chunks:
                    output_chunk = model(data_chunk)
                    output_chunks.append(output_chunk)
                output = torch.cat(output_chunks, dim=0)

                # Remove the variables
                del output_chunk, output_chunks, data_chunk, data_chunks

                # Update aggregator with this batch’s ground truth + predictions
                target = convert_output(target, sed_threshold=sed_threshold)
                output = convert_output(output, sed_threshold=sed_threshold)

                seld_metric.update(gt=target, pred=output)

                progress.update(task, advance=1)

    # Compute the final metrics across all batches
    ER, F, LE, LR = seld_metric.compute()

    del data, target, output
    torch.cuda.empty_cache()
    return ER, F, LE, LR


if __name__ == "__main__":

    # Device configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Get the test dataset
    dataset = seldDatabase()
    test_data = dataset.get_split("test")
    test_batch_size = test_data["test_batch_size"]
    test_dataset = seldDataset(db_data=test_data)

    sample_x, sample_y = test_dataset[0]
    data_in = sample_x.shape
    data_out = sample_y.shape

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=test_batch_size, shuffle=False,
                                 num_workers=0, drop_last=False,
                                 pin_memory=True, prefetch_factor=2)

    model_weight_loc = "./model_weights/270125_1406_dsc_block_model.h5"
    model = ResNet(in_feat_shape=data_in,
                   out_feat_shape=data_out,
                   use_dsc=True, 
                   btn_dsc=False).to(device)
    model.load_state_dict(torch.load(model_weight_loc, map_location='cpu'))

    best_er, best_f1, best_le, best_lr, best_seld, best_threshold = 9999, 0., 180., 0., 9999, 0

    for sed_thres in np.arange(0.1, 1.0, 0.1):
        er, f1, le, lr = inference(test_dataloader, 
                                model=model, 
                                device=device, 
                                nb_batches=len(test_dataloader),
                                sed_threshold=sed_thres)

        e_seld = (er + (1-f1) + (le/180) + (1-lr))/4

        if e_seld < best_seld:
            best_er, best_f1, best_le, best_lr, best_seld, best_threshold = er, f1, le, lr, e_seld, sed_thres

        print("[{:.1f}] ER/F1/LE/LR/SELD: {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.3f}".format(sed_thres, er, f1, le, lr, e_seld))

    print("\nBest Threshold: {:.1f}".format(best_threshold))

    er, f1, le, lr = inference(test_dataloader, 
                                model=model, 
                                device=device, 
                                nb_batches=len(test_dataloader),
                                sed_threshold=best_threshold)

    e_seld = (er + (1-f1) + (le/180) + (1-lr))/4
    print("Best ER/F1/LE/LR/SELD: {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.3f}".format(er, f1, le, lr, e_seld))