__kernel
void func(__global float *X,
          __global float *Y)
{
    // Get the work-item's unique ID
    int idx=get_global_id(0);
    float dn = X[idx]*X[idx] + 1.L;
    Y[idx] = 4.L  / dn;
}
