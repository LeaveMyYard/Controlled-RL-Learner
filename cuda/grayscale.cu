__global__ void dot(int *result, int *a)
{
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    //int threadId = blockId * blockDim.x + threadIdx.x;

    int r = a[blockId*3    ];
    int g = a[blockId*3 + 1];
    int b = a[blockId*3 + 2];

    result[blockId] = (r+g+b)/3;
}