// Take A as a bin value and place it into a histogram bin
kernel void histogram(global const uchar* A, global int* H) {
	// Assumes that H has been initialised to 0 from writing buffer with 0's
	int id = get_global_id(0);

	// Increment bin indexes in serial
	atomic_inc(&H[A[id]]);
}

//requires additional buffer B to avoid data overwrite
//final result stored in B
kernel void scan_hs(global int* input, global int* output) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	printf("%i, ", id);

	global int* temp;
	for (int stride=1;stride<N; stride*=2) {
		output[id] = input[id];
		if (id >= stride) {
			output[id] += input[id - stride];
		}

		//sync the step
		barrier(CLK_GLOBAL_MEM_FENCE);
		//swap A & B between steps
		temp = input;
		input = output;
		output = temp;
	}
}