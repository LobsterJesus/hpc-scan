__kernel void naiveScan(
	__global const int * restrict input,
	__global int *output,
	const int n)
{
	__global int *temp = output;
	int thid = get_local_id(0);
	int pout = 0;
	int pin = 1;
			
	temp[pout * n + thid] = (thid > 0) ? input[thid - 1] : 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int offset = 1; offset < n; offset *= 2) { 
		pout = 1 - pout;
		pin = 1 - pout;
		
		temp[pout * n + thid] = temp[pin * n + thid];

		if (thid >= offset)
			temp[pout * n + thid] += temp[pin * n + thid - offset];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	output[thid] = temp[pout * n + thid];
}