//a simple OpenCL kernel which copies all pixels from A to B
kernel void identity(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);

	// Simple copy from A to B
	B[id] = A[id];
}

kernel void filter_r(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0)/3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

	printf("Channel: %i\n", colour_channel);

	// Exclusively copy red colour channels over to our output image's buffer
	if (colour_channel == 0) {
		B[id] = A[id];
	}
	else {
		B[id] = 0;
	}

	//this is just a copy operation, modify to filter out the individual colour channels
	//B[id] = A[id];
}

// Invert the current pixel intensity value for each pixel in a CImg array, currently also does histogram!
kernel void invert(global const uchar* A, global uchar* B, global int* H) {
	int id = get_global_id(0);

	////printf("%i, ", A[id]);
	
	// Take a max value of 255 and subtract a value (0-255) from it to get the inverted value, e.g 255 - 0 = 255; 255 - 100 = 155
	B[id] = 255 - A[id];

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index
	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

//simple ND identity kernel
kernel void identityND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	B[id] = A[id];
}

//2D averaging filter
kernel void avg_filterND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	uint result = 0;

	//simple boundary handling - just copy the original pixel
	if ((x == 0) || (x == width-1) || (y == 0) || (y == height-1)) {
		result = A[id];	
	} else {
		for (int i = (x - 1); i <= (x + 1); i++) {
			for (int j = (y - 1); j <= (y + 1); j++) {
				result += A[i + j*width + c*image_size];
			}
		}

		result /= 9;
	}

	B[id] = (uchar)result;
}

//2D 3x3 convolution kernel
kernel void convolutionND(global const uchar* A, global uchar* B, constant float* mask) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	float result = 0;

	//simple boundary handling - just copy the original pixel
	if ((x == 0) || (x == width-1) || (y == 0) || (y == height-1)) {
		result = A[id];	
	} else {
		for (int i = (x-1); i <= (x+1); i++)
		for (int j = (y-1); j <= (y+1); j++) 
			result += A[i + j*width + c*image_size]*mask[i-(x-1) + j-(y-1)];
	}

	B[id] = (uchar)result;
}

// Serialized histogram implementation
kernel void hist_simple(global const int* A, global int* H) {
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index
	printf("Bin Index: %i", bin_index);

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

//requires additional buffer B to avoid data overwrite
//final result stored in B
kernel void scan_hs(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;
	for (int stride=1;stride<N; stride*=2) {
		B[id] = A[id];
		if (id >= stride) {
			B[id] += A[id - stride];
		}

		//sync the step
		barrier(CLK_GLOBAL_MEM_FENCE);
		//swap A & B between steps
		C = A; A = B; B = C;
		printf("%i", B);
	}
}