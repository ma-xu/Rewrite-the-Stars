r"""
# Use case:
# CUDA_VISIBLE_DEVICES=0 python3 benchmark_onnx.py --model {model-name} --input-size 3 244 244 --benchmark_cpu

Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""
import argparse
import os
import sys
import csv
import glob
import json
import time
import logging
import numpy as np
import torch
import torch.nn.parallel

import timm
from timm.models import create_model
from timm.data import resolve_data_config
from timm.utils import setup_default_logging
import onnx
import onnxruntime
import cpuinfo
import tensorrt
from fvcore.nn import FlopCountAnalysis
from thop import profile, clever_format
import starnet

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('benchmark')

parser = argparse.ArgumentParser(description='ONNX benchmark')
parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# speed benchmark
parser.add_argument('--nwarmup', default=50, type=int, help='warm up iterations')
parser.add_argument('--nruns', default=400, type=int,
                    help='Average benchmark speed over {nruns} iterations')
parser.add_argument('--benchmark_bs', default=1, type=int,
                    help='The batch size for speed benchmark')
parser.add_argument('--comments', default="", type=str,
                    help='Any string comments for this script')
parser.add_argument('--results_file', default='debug.csv', type=str, metavar='FILENAME',
                    help='Output csv file for benchmark results (summary)')
parser.add_argument('--intra_op_num_threads', default=1, type=int,
                    help='threads for onnxruntime test, works for gpu, cpu and pytorch')
parser.add_argument('--benchmark_cpu', default=False, action='store_true',
                    help='If we should benchmark the inference speed on cpu')
parser.add_argument('--opset_version', default=12, type=int, help='opset version')


def validate(args):
    # create model
    model = create_model(args.model, pretrained=args.pretrained)
    model.eval()
    model_params = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, model_params))

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)

    # export onnx
    dummy_input = torch.randn(args.benchmark_bs, data_config['input_size'][0], data_config['input_size'][1],
                              data_config['input_size'][2], requires_grad=True)
    if not os.path.exists("onnx_models"):
        os.makedirs("onnx_models")
    torch.onnx.export(model,
                      dummy_input,
                      os.path.join("onnx_models", args.model + ".onnx"),
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=args.opset_version,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}
                                    }
                      )  # the model's output names
    _logger.info(f"===> Successfully export onnx")

    model = model.cuda()

    model.eval()
    flops_input = torch.randn((1,) + tuple(data_config['input_size'])).cuda()
    fvcore_flops = FlopCountAnalysis(model, flops_input)
    # _logger.info(f"flops is: {flops.total()}")
    ### update: the profile lib calculate error param numbers, use ours.
    model_macs, _ = profile(model, inputs=(flops_input,), verbose=False)
    model_flops, model_macs, model_params = clever_format([fvcore_flops.total(), model_macs, model_params], "%.3f")
    _logger.info(f"flops is: {model_flops}, macs is: {model_macs}, params is: {model_params}")

    ### benchmark speed ####
    _logger.info('\n===> Start benchmarking speed\n')
    input = torch.randn((args.benchmark_bs,) + tuple(data_config['input_size'])).cuda()
    # benchmark
    # 1: speed benchmark: PyTorch
    _logger.info(f"\n===> Warm up {args.nwarmup} iterations for Pytorch speed benchmarking ...")
    with torch.no_grad():
        for _ in range(args.nwarmup):
            features = model(input)
    torch.cuda.synchronize()
    timings = []
    _logger.info(f"\n===> Benchmark {args.nruns} iterations for Pytorch speed benchmarking ...")
    with torch.no_grad():
        for i in range(1, args.nruns + 1):
            torch.cuda.synchronize()
            start_time = time.time()
            model(input)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
        pytorch_speed = np.mean(timings) * 1000
    _logger.info('\n===> Benchmarking Pytorch speed, avgerage batch time %.2f ms\n\n' % (pytorch_speed))

    # 2: speed benchmark: ONNX GPU
    input = input.detach().cpu().numpy() if input.requires_grad else input.cpu().numpy()
    _logger.info(f"\n===> Warm up {args.nwarmup} iterations for ONNX GPU speed benchmarking ...")
    providers = [('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    })]
    opts = onnxruntime.SessionOptions()
    opts.enable_profiling = True  # if profiling the details
    if not os.path.exists("./model_profiles/"):
        os.makedirs("./model_profiles/")
    opts.profile_file_prefix = "./model_profiles/" + args.model
    opts.intra_op_num_threads = args.intra_op_num_threads
    session = onnxruntime.InferenceSession(os.path.join("onnx_models", args.model + ".onnx")
                                           , providers=providers, sess_options=opts)
    # IOBinding
    input_names = session.get_inputs()[0].name
    output_names = session.get_outputs()[0].name
    io_binding = session.io_binding()
    io_binding.bind_cpu_input(input_names, input)
    io_binding.bind_output(output_names, 'cuda')
    # for profiling
    session.run_with_iobinding(io_binding)
    profile_file = session.end_profiling()
    print(f"\n===> Profiling file name is: {profile_file}")
    for _ in range(args.nwarmup):
        session.run_with_iobinding(io_binding)
    torch.cuda.synchronize()
    timings = []
    _logger.info(f"\n===> Benchmark {args.nruns} iterations for ONNX GPU speed benchmarking ...")
    with torch.no_grad():
        for i in range(1, args.nruns + 1):
            torch.cuda.synchronize()
            start_time = time.time()
            session.run_with_iobinding(io_binding)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
        onnx_gpu_speed = np.mean(timings) * 1000
    _logger.info('\n===> Benchmarking ONNX GPU speed, avgerage batch time %.2f ms\n\n' % (onnx_gpu_speed))

    memory_allocated = torch.cuda.memory_allocated(device=next(model.parameters()).device)
    max_memory_allocated = torch.cuda.max_memory_allocated(device=next(model.parameters()).device)
    torch.cuda.reset_peak_memory_stats()
    _logger.info(f"ONNX memory_allocated: {memory_allocated}, max_memory_allocated: {max_memory_allocated}")

    del session

    # 3: speed benchmark: ONNX CPU
    onnx_cpu_speed = 0.
    if args.benchmark_cpu:
        _logger.info(f"\n===> Warm up {args.nwarmup} iterations for ONNX CPU speed benchmarking ...")
        providers = ['CPUExecutionProvider']
        opts = onnxruntime.SessionOptions()
        # opts.enable_profiling = True
        opts.intra_op_num_threads = args.intra_op_num_threads
        session = onnxruntime.InferenceSession(os.path.join("onnx_models", args.model + ".onnx"),
                                               providers=providers, sess_options=opts)
        for _ in range(args.nwarmup):
            session.run([], {'input': input})
        torch.cuda.synchronize()
        timings = []
        _logger.info(f"\n===> Benchmark {args.nruns} iterations for ONNX CPU speed benchmarking ...")
        with torch.no_grad():
            # reduce nruns to reduce waiting time for cpu since it is really stable.
            for i in range(1, args.nruns // 5 + 1):
                start_time = time.time()
                session.run([], {'input': input})
                end_time = time.time()
                timings.append(end_time - start_time)
            onnx_cpu_speed = np.mean(timings) * 1000
        _logger.info('\n===> Benchmarking ONNX CPU speed, avgerage batch time %.2f ms\n\n' % (onnx_cpu_speed))
        del session

    log_results = {
        #  model related logs
        "model_model": args.model,
        "model_params": model_params,
        "model_flops": model_flops,
        "model_macs": model_macs,
        "model_memory": clever_format(memory_allocated, "%.3f"),
        # data related
        "data_input_size": data_config['input_size'],
        # benchmark related
        "benchmark_git_commit_id": get_git_commit_id(),
        "benchmark_date": time.strftime('%Y-%m-%d:%H:%M:%S', time.localtime()),
        "benchmark_pytorch_latency": "{:.3f}".format(pytorch_speed),
        "benchmark_onnx_gpu_latency": "{:.3f}".format(onnx_gpu_speed),
        "benchmark_onnx_cpu_latency": "{:.3f}".format(onnx_cpu_speed),
        "benchmark_bs": args.benchmark_bs,
        "benchmark_nwarmup": args.nwarmup,
        "benchmark_nruns": args.nruns,
        #  system related logs
        "system_verision_python": sys.version.replace('\n', ''),
        "system_verision_pytorch": torch.__version__,
        "system_verision_timm": timm.__version__,
        "system_verision_cuda": torch.version.cuda,
        "system_verision_cudnn": torch.backends.cudnn.version(),
        "system_verision_onnx": onnx.__version__,
        "system_verision_onnxruntime": onnxruntime.__version__,
        "system_verision_tensorrt": tensorrt.__version__,
        "system_gpu_name": torch.cuda.get_device_name(0),
        "system_cpu_arch": cpuinfo.get_cpu_info()["arch"],
        "system_cpu_brand_raw": cpuinfo.get_cpu_info()["brand_raw"],
        "opset_version": args.opset_version,
        "others_comments": args.comments,
        "profile_file": profile_file
    }
    return log_results


def get_git_commit_id():
    try:
        import subprocess
        cmd_out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        return cmd_out
    except:
        # indicating no git found.
        return "0000000"


def main():
    setup_default_logging()
    args = parser.parse_args()
    results = validate(args)
    # output results in JSON to stdout w/ delimiter for runner script
    print(f'\n===> Benchmark result:\n{json.dumps(results, indent=4)}')
    try:
        write_results(args.results_file, results)
        print(f"Successfully write results to {args.results_file}")
    except:
        print(f"Write CSV error")


def write_results(results_file, results):
    csv_isfile = os.path.isfile(results_file)
    with open(results_file, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results.keys())
        if not csv_isfile:
            writer.writeheader()
        writer.writerows([results])
        csvfile.flush()
        csvfile.close()


if __name__ == '__main__':
    main()
