ncu --metrics sm__inst_executed_pipe_tensor,sm__inst_executed_pipe_tensor_op_hmma,sm__inst_executed_pipe_tensor_op_imma,sm__pipe_tensor_cycles_active,sm__pipe_tensor_op_hmma_cycles_active,sm__pipe_tensor_op_imma_cycles_active \
    --print-summary per-kernel --profile-from-start off \
    gemm_cuda_bench 2048 2048 2048 -I 10 -M fp16 -A fp16

ncu --metrics sm__inst_executed_pipe_tensor,sm__inst_executed_pipe_tensor_op_hmma,sm__inst_executed_pipe_tensor_op_imma,sm__pipe_tensor_cycles_active,sm__pipe_tensor_op_hmma_cycles_active,sm__pipe_tensor_op_imma_cycles_active \
    --print-summary per-kernel --profile-from-start off \
    gemm_cuda_bench 2048 2048 2048 -I 10 -M fp16 -A fp16 --cudacoresonly

ncu --metrics sm__inst_executed_pipe_tensor,sm__inst_executed_pipe_tensor_op_hmma,sm__inst_executed_pipe_tensor_op_imma,sm__pipe_tensor_cycles_active,sm__pipe_tensor_op_hmma_cycles_active,sm__pipe_tensor_op_imma_cycles_active \
    --print-summary per-kernel --profile-from-start off \
    gemm_cuda_bench 2048 2048 2048 -I 10 -M fp16 -A fp32

ncu --metrics sm__inst_executed_pipe_tensor,sm__inst_executed_pipe_tensor_op_hmma,sm__inst_executed_pipe_tensor_op_imma,sm__pipe_tensor_cycles_active,sm__pipe_tensor_op_hmma_cycles_active,sm__pipe_tensor_op_imma_cycles_active \
    --print-summary per-kernel --profile-from-start off \
    gemm_cuda_bench 2048 2048 2048 -I 10 -M fp16 -A fp32 --cudacoresonly

ncu --metrics sm__inst_executed_pipe_tensor,sm__inst_executed_pipe_tensor_op_hmma,sm__inst_executed_pipe_tensor_op_imma,sm__pipe_tensor_cycles_active,sm__pipe_tensor_op_hmma_cycles_active,sm__pipe_tensor_op_imma_cycles_active \
    --print-summary per-kernel --profile-from-start off \
    gemm_cuda_bench 2048 2048 2048 -I 10 -M fp32 -A fp32

ncu --metrics sm__inst_executed_pipe_tensor,sm__inst_executed_pipe_tensor_op_hmma,sm__inst_executed_pipe_tensor_op_imma,sm__pipe_tensor_cycles_active,sm__pipe_tensor_op_hmma_cycles_active,sm__pipe_tensor_op_imma_cycles_active \
    --print-summary per-kernel --profile-from-start off \
    gemm_cuda_bench 2048 2048 2048 -I 10 -M fp32 -A fp32 --cudacoresonly

ncu --metrics sm__inst_executed_pipe_tensor,sm__inst_executed_pipe_tensor_op_hmma,sm__inst_executed_pipe_tensor_op_imma,sm__pipe_tensor_cycles_active,sm__pipe_tensor_op_hmma_cycles_active,sm__pipe_tensor_op_imma_cycles_active \
    --print-summary per-kernel --profile-from-start off \
    gemm_cuda_bench 2048 2048 2048 -I 10 -M fp64 -A fp64

ncu --metrics sm__inst_executed_pipe_tensor,sm__inst_executed_pipe_tensor_op_hmma,sm__inst_executed_pipe_tensor_op_imma,sm__pipe_tensor_cycles_active,sm__pipe_tensor_op_hmma_cycles_active,sm__pipe_tensor_op_imma_cycles_active \
    --print-summary per-kernel --profile-from-start off \
    gemm_cuda_bench 2048 2048 2048 -I 10 -M fp64 -A fp64 --cudacoresonly

ncu --metrics sm__inst_executed_pipe_tensor,sm__inst_executed_pipe_tensor_op_hmma,sm__inst_executed_pipe_tensor_op_imma,sm__pipe_tensor_cycles_active,sm__pipe_tensor_op_hmma_cycles_active,sm__pipe_tensor_op_imma_cycles_active \
    --print-summary per-kernel --profile-from-start off \
    gemm_cuda_bench 2048 2048 2048 -I 10 -M int8 -A int8

ncu --metrics sm__inst_executed_pipe_tensor,sm__inst_executed_pipe_tensor_op_hmma,sm__inst_executed_pipe_tensor_op_imma,sm__pipe_tensor_cycles_active,sm__pipe_tensor_op_hmma_cycles_active,sm__pipe_tensor_op_imma_cycles_active \
    --print-summary per-kernel --profile-from-start off \
    gemm_cuda_bench 2048 2048 2048 -I 10 -M int8 -A int8 --cudacoresonly

