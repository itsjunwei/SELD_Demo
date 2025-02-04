import numpy as np
import torch.backends
import torch
from torch.utils.data import DataLoader
from manual_dataset import *
from rich.progress import Progress
from models import ResNet
from utilities import *

# Misc utility functions
def to_numpy(tensor):
    """Convert the feature tensor into np.ndarray format for the ONNX model to run 

    Inputs
        tensor (PyTorch Tensor) : input PyTorch feature tensor of any shape 

    Returns
        tensor (PyTorch Tensor) : The same tensor, but in np.ndarray format to input into ONNX model
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def inference(data_generator,  model, device, nb_batches=1000, sed_threshold=0.5, is_onnx=False):
    if is_onnx is False:
        model.eval()

    # Create an instance of the aggregator
    seld_metric = SELDMetricsAzimuth(n_classes=3, azimuth_threshold=20, sed_threshold=sed_threshold, out_class=True)

    with Progress(transient=True) as progress:
        task = progress.add_task("[green]Inference: ", total=nb_batches)

        with torch.no_grad():
            for data, target in data_generator:

                data , target = data.to(device), target.to(device)

                # Split the data into managable chunks, in this case 32 items
                data_chunks = torch.split(data, 32)
                output_chunks = []
                for data_chunk in data_chunks:
                    if is_onnx: # ONNX Inference
                        # input_tensor = torch.from_numpy(data).type(torch.FloatTensor).unsqueeze(0)
                        inputs = {input_names: to_numpy(data_chunk)}
                        pred = model.run(None, inputs)
                        output_chunk = torch.from_numpy(pred[0])
                    else:
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
    dataset = seldDatabase(feat_label_dir="./feat_label_2fps")
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
    
    model_weight_loc = "./model_weights/030225_0913_btndsc_2fps_model.h5"
    # model_weight_loc = './onnx_models/030225_0913_btndsc_2fps_model.onnx'

    if model_weight_loc.endswith(".h5"):
        # Loading model via PyTorch weights
        onnx_model = False
        model = ResNet(in_feat_shape=data_in,
                       out_feat_shape=data_out,
                       use_dsc=False, 
                       btn_dsc=True,
                       fps=2).to(device)
        model.load_state_dict(torch.load(model_weight_loc, map_location='cpu'))
        model.eval()
    else:
        # Loading ONNX Inference Model
        onnx_model = True
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        model = ort.InferenceSession(model_weight_loc, sess_options=sess_options)
        input_names = model.get_inputs()[0].name

    best_er, best_f1, best_le, best_lr, best_seld, best_threshold = 9999, 0., 180., 0., 9999, 0

    for sed_thres in np.arange(0.1, 1.0, 0.1):
        er, f1, le, lr = inference(test_dataloader, 
                                model=model, 
                                device=device, 
                                nb_batches=len(test_dataloader),
                                sed_threshold=sed_thres,
                                is_onnx=onnx_model)

        e_seld = (er + (1-f1) + (le/180) + (1-lr))/4

        if e_seld < best_seld:
            best_er, best_f1, best_le, best_lr, best_seld, best_threshold = er, f1, le, lr, e_seld, sed_thres

        print("[{:.1f}] ER/F1/LE/LR/SELD: {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.3f}".format(sed_thres, er, f1, le, lr, e_seld))

    print("\nModel used: {}".format("ONNX" if onnx_model else "PyTorch"))
    print("Best Threshold: {:.1f}".format(best_threshold))

    er, f1, le, lr = inference(test_dataloader, 
                                model=model, 
                                device=device, 
                                nb_batches=len(test_dataloader),
                                sed_threshold=best_threshold,
                                is_onnx=onnx_model)

    e_seld = (er + (1-f1) + (le/180) + (1-lr))/4
    print("Best ER/F1/LE/LR/SELD: {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.3f}".format(er, f1, le, lr, e_seld))