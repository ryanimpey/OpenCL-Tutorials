// Take A as a bin value and place it into a histogram bin
kernel void histogram(global const uchar* A, global int* H) {
	// Assumes that H has been initialised to 0 from writing buffer with 0's
	int id = get_global_id(0);

	// Increment bin indexes in serial
	atomic_inc(&H[A[id]]);
}

//simple exclusive serial scan based on atomic operations - sufficient for small number of elements
kernel void scan_add_atomic(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id + 1; i < N; i++)
		atomic_add(&B[i], A[id]);
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

//a double-buffered version of the Hillis-Steele inclusive scan
//requires two additional input arguments which correspond to two local buffers
kernel void scan_add(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int* scratch_3;//used for buffer swap

	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	//copy the cache to output array
	B[id] = scratch_1[lid];
}

// TODO: Finish this
kernel void bin_normalise(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);

	// Take a max value of 255 and subtract a value (0-255) from it to get the inverted value, e.g 255 - 0 = 255; 255 - 100 = 155
	int inverted_value = 255 - A[id];
	B[id] = inverted_value;
}